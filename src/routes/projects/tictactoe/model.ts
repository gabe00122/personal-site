import * as tf from "@tensorflow/tfjs";
import type { GameState } from "./game";


interface ActionResult {
    action: number;
    preferences: number[];
    value: number;
}


export class ModelWrapper {
    #model: tf.GraphModel<string>;
    
    constructor(model: tf.GraphModel<string>) {
        this.#model = model;
    }

    async act(game: GameState): Promise<ActionResult> {
        const obs = ModelWrapper.#encodeObservation(game);
        const mask = ModelWrapper.#encodeMask(game);
        
        const [value, logits] = await this.#model.predictAsync({xs_0: obs, xs_1: mask}) as [tf.Tensor, tf.Tensor];

        const preferences = tf.softmax(logits);

        const jsPreferences = await preferences.data();
        const jsValue = await value.data();

        let max = Number.MIN_VALUE;
        let greedyIndex = -1;

        for (let i = 0; i < jsPreferences.length; i++) {
            if (jsPreferences[i] > max) {
                greedyIndex = i;
                max = jsPreferences[i];
            }
        }

        return {
            action: greedyIndex,
            preferences: [...jsPreferences],
            value: jsValue[0],
        };
    }

    static #encodeObservation(game: GameState): tf.Tensor<tf.Rank> {
        const observationList = game.board.flatMap((cell) => {
            if (cell === undefined) {
                return [0, 1, 0];
            } else if (cell  === game.activePlayer){
                return [1, 0, 0];
            } else {
                return [0, 0, 1];
            }
        });

        return tf.tensor(observationList);
    }

    static #encodeMask(game: GameState): tf.Tensor<tf.Rank> {
        const observationList = game.board.map((cell) => cell === undefined);
        return tf.tensor(observationList);
    }
}

