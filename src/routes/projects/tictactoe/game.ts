export type Player = 'X' | 'O';
export type Cell = Player | undefined;

export interface GameState {
	board: Cell[];
	activePlayer: Player;
	result: 'ongoing' | 'draw' | 'won';
}

export const initialGameState: GameState = {
	board: [
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined
	],
	activePlayer: 'X',
	result: 'ongoing'
};

export function turn(state: GameState, action: number): GameState {
	const board = [...state.board];
	const activePlayer = state.activePlayer;

	board[action] = state.activePlayer;

	return {
		board,
		activePlayer: nextPlayer(activePlayer),
		result: checkWon(board, activePlayer) ? 'won' : checkOver(board) ? 'draw' : 'ongoing'
	};
}

function nextPlayer(player: Player): Player {
	if (player === 'X') {
		return 'O';
	} else {
		return 'X';
	}
}

function checkWon(board: Cell[], activePlayer: Player): boolean {
	const patterns = [
		[0, 1, 2],
		[3, 4, 5],
		[6, 7, 8],
		[0, 3, 6],
		[1, 4, 7],
		[2, 5, 8],
		[0, 4, 8],
		[6, 4, 2]
	];

	outer: for (const pattern of patterns) {
		for (const index of pattern) {
			if (board[index] !== activePlayer) {
				continue outer;
			}
		}
		return true;
	}

	return false;
}

function checkOver(board: Cell[]): boolean {
	return board.every((cell) => cell !== undefined);
}
