# Function to check if it is safe to place num at mat[row][col]
def check_constraint(mat, row, col, num):
    
    # Check if num exists in the row
    for x in range(9):
        if mat[row][x] == num:
            return False

    # Check if num exists in the col
    for x in range(9):
        if mat[x][col] == num:
            return False

    # Check if num exists in the 3x3 sub-matrix
    startRow = row - (row % 3)
    startCol = col - (col % 3)

    for i in range(3):
        for j in range(3):
            if mat[i + startRow][j + startCol] == num:
                return False

    return True

# Function to solve the Sudoku problem
def recursive_solve(mat, row, col):
    # base case: Reached nth column of the last row
    if row == 8 and col == 9:
        return True

    # If last column of the row go to the next row
    if col == 9:
        row += 1
        col = 0

    # If cell is already occupied then move forward
    if mat[row][col] != 0:
        return recursive_solve(mat, row, col + 1)

    for num in range(1, 10):
        
        # If it is safe to place num at current position
        if check_constraint(mat, row, col, num):
            mat[row][col] = num
            if recursive_solve(mat, row, col + 1):
                return True
            mat[row][col] = 0

    return False

def solve_sudoku(mat):
    if not recursive_solve(mat, 0, 0):
        raise RuntimeError("Puzzle is unsolvable — check digit recognition output")
    return mat