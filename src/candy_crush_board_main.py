import candy_crush_board

config_file = '../config/config1.txt'
board = candy_crush_board.CandyCrushBoard(config_file=config_file)
print(board.get_board())
