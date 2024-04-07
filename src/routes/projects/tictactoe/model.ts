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

        const jsPreferences = [...await preferences.data()];
        const jsValue = (await value.data())[0];

        const action = ModelWrapper.#samplePreference(jsPreferences);

        return {
            action,
            preferences: jsPreferences,
            value: jsValue,
        };
    }

    static #samplePreference(preferences: number[]): number {
        const random = Math.random();
        let sum = 0;

        for (let i = 0; i < preferences.length; i++) {
            sum += preferences[i];
            if (sum >= random) {
                return i;
            }
        }
        return preferences.length - 1;
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
        
        //It's always the models turn
        observationList.push(1.0);
        observationList.push(0.0);

        return tf.tensor(observationList);
    }

    static #encodeMask(game: GameState): tf.Tensor<tf.Rank> {
        const observationList = game.board.map((cell) => cell === undefined);
        return tf.tensor(observationList);
    }
}

