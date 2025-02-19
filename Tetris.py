import random
import matplotlib.pyplot as plt
import numpy as np
import joblib

nn = None
def render_tetris_grid(grid):
    rows, cols = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(cols, rows))

    # Create a colored grid
    for r in range(rows):
        for c in range(cols):
            color = tuple(grid[r][c] for x in range(3))
            ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, color=color, ec="black"))

    # Set limits and remove axes
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.show()




# global variables
col = 10  # 10 columns
row = 20  # 20 rows

# shapes formats
S = [[
    '.00',
    '00.'],
     [
    '0.','00','.0']]

Z = [['00.', '.00'],
     [
    '.0','00','0.']]

I = [[
    '0','0','0','0'],
     [
    '0000']]

O = [['00','00']]

J = [['0..',
    '000'],
     [
    '00',
    '0.',
    '0.'],
     [
    '000',
    '..0'],
     [
    '.0',
    '.0',
    '00']]

L = [[
    '..0',
    '000'],
     [
    '0.',
    '0.',
    '00'],
     [
    '000',
    '0..'],
     [
    '00'
    '.0',
    '.0']]

T = [[
    '.0.',
    '000'],
     [
    '0.',
    '00',
    '0.'],
     [
    '000',
    '.0.'],
     [
    '.0',
    '00',
    '.0']]

# index represents the shape
shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]


# class to represent each of the pieces
class Piece(object):
    def __init__(self, x, y, shape):
      self.x = x
      self.y = y
      self.shape = shape
      self.color = shape_colors[shapes.index(shape)]  # choose color from the shape_color list
      self.rotation = 0  # chooses the rotation according to index
    

# initialise the grid
def create_grid(locked_pos={}):
    grid = [[(0,0,0) for x in range(col)] for y in range(row)]  # grid represented rgb tuples

    for y in range(row):
      for x in range(col):
        if (x, y) in locked_pos:
            color = locked_pos[(x, y)]
            grid[y][x] = color

    return grid


def convert_shape_format(piece):
    positions = []
    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(shape_format):
      row = list(line)
      for j, column in enumerate(row):
        if column == '0':
            positions.append((piece.x + j, piece.y + i))

    for i, pos in enumerate(positions):
      positions[i] = (pos[0], pos[1])

    return positions


def valid_space(piece, grid):
    accepted_pos = [[(x, y) for x in range(col) if grid[y][x] == 0] for y in range(row)]
    accepted_pos = [x for item in accepted_pos for x in item]

    formatted_shape = convert_shape_format(piece)

    for pos in formatted_shape:
      if pos not in accepted_pos:
        if pos[1] >= 0:
            return False
    return True


def check_lost(positions):
    for pos in positions:
      x, y = pos
      if y < 1:
        return True
    return False


def get_shape():
    return Piece(0, 4, random.choice(shapes))


def clear_rows(grid):
    increment = 0
    for i in range(len(grid) - 1, -1, -1):
        grid_row = grid[i]
        if 0 not in grid_row:
            increment += 1
            index = i
        else:
           grid[i+increment] = grid[i]
    

    return increment


def update_score(new_score, filepath='./highscore.txt'):
    score = get_max_score(filepath)

    with open(filepath, 'w') as file:
      if new_score > score:
        file.write(str(new_score))
      else:
        file.write(str(score))


def get_max_score(filepath='./highscore.txt'):
    with open(filepath, 'r') as file:
      lines = file.readlines()
      score = int(lines[0].strip())

    return score


def main():
    grid = np.array([[0 for x in range(col)] for y in range(row)])

    run = True
    #next_piece = get_shape()
    score = 0
   # grid_hist = []
    while run:
        print(nn.predict([grid.flatten()]))
        current_piece = get_shape()

        #flattened_grid = [item for sublist in grid for item in sublist]
        #grid_hist.append(([1 if i != (0, 0, 0) else 0 for i in flattened_grid], score))
        valid_states = []
        for i in range(col):
            for j in range(row):
                for k in range(len(current_piece.shape)):
                    current_piece.x = i
                    current_piece.y = j
                    current_piece.rotation = k
                    if valid_space(current_piece, grid):
                        current_piece.y += 1
                        if not valid_space(current_piece, grid):
                            valid_states.append((i, j, k))
        if len(valid_states) == 0:
            #render_tetris_grid(grid)
            print(score)
            return score
            return [(i, score-j) for i, j in grid_hist]
        predicted_scores = []
        for current_piece.x, current_piece.y, current_piece.rotation in valid_states:
            piece_pos = convert_shape_format(current_piece)
            grid_copy = grid.copy()
            for pos in piece_pos:
                x, y = pos
                if y >= 0:
                    grid_copy[y][x] = 1
            predicted_scores.append(clear_rows(grid_copy)*10)
            predicted_scores[-1] += nn.predict([grid_copy.flatten()])[0]
        max_index = np.argmax(predicted_scores)
        
        current_piece.x, current_piece.y, current_piece.rotation = max(valid_states, key=lambda x: x[1])

        change_piece = True
        #print(tuple((current_piece.x, current_piece.y, current_piece.rotation)))
        piece_pos = convert_shape_format(current_piece)

        for i in range(len(piece_pos)):
            x, y = piece_pos[i]
            if y >= 0:
                grid[y][x] = 1
        score += clear_rows(grid) * 10

        '''if check_lost(locked_positions):
            print("checklost")
            run = False
            render_tetris_grid(grid)'''




if __name__ == '__main__':
    nn = joblib.load('tetris_nn_model.pkl')
    #print(nn.predict([[0 for i in range(200)]]))
    main()
    '''for i in range(100):
        score = main()
        with open('tetris_scores.txt', 'a') as f:
            f.write(f"Game {i+1}: {score}\n")'''
