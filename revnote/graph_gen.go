package revnote

// '1'로 연결되어 있는 집합(섬)을 찾는 문제
// bfs로 모든 셀 방문해서 값 변경하면서 카운팅
func numIslands(grid [][]byte) int {

	ret := 0
	for x := range grid { // 이중for문으로 모든 셀에서 시작하기
		for y := range grid[x] {
			if grid[x][y] == '1' { // '1'이 발견되면 카운팅
				ret++
				nbfs(grid, x, y)
			}
		}
	}

	return ret
}

func nbfs(grid [][]byte, x, y int) {
	if x < 0 || y < 0 || x >= len(grid) || y >= len(grid[x]) || grid[x][y] == '0' { // 종료조건
		// 조건을 x > len(grid) || y > len(grid[x]) 이렇게 줘도
		// out of range 에러가 안 나는데, 이미 마지막 셀에서 값을 'O'로 바꿨기 때문
		return
	}

	grid[x][y] = '0' // 중복 방문을 막기 위해 0으로 변경

	// 특정 x, y에서 4방면을 모두 방문하기 시작
	// 왜냐면 1로 연결된 곳까지 탐색을 해야하니까
	nbfs(grid, x-1, y)
	nbfs(grid, x+1, y)
	nbfs(grid, x, y-1)
	nbfs(grid, x, y+1)
}

// 4면이 x로 둘러 쌓여있지 않은 o만 x로 바꿔주면 됨.
// 이건 3면은 x로 둘러쌓였으면서 배열 가장 바깥에 위치한 o와
// 걔랑 연결된 o는 바꾸면 안 된다는 뜻
func solve(board [][]byte) {

	// 한 방에 해결은 못 하고, 두 번에 걸쳐서 해야함.
	// 1) 일단 맨 바깥에 위치한 o와 걔랑 연결된 o만 다른 문자열로 바꿔줌.
	// 2) 다시 이중for문 돌면서 다른 문자열을 전부 o로, 그리고 o는 x로 바꿔줌. (순서 상관x)

	for x := range board {
		for y := range board[x] {
			// 맨 바깥에 위치한 O를 *로 바꾸기 시작
			if x == 0 || y == 0 || x == len(board)-1 || y == len(board[x])-1 {
				if board[x][y] == 'O' {
					sbfs(x, y, board)
				}
			}
		}
	}

	// 맨 바깥에 위치한 o, 그리고 그거랑 연결된 o만 *로 바뀌고,
	// x에 둘러쌓인 o는 안 바뀐 상태로 board가 넘어오게 됨.
	// 따라서, *을 o로 바꿔주고, o를 x로 바꿔주면 문제가 해결됨.
	for x := range board {
		for y := range board[x] {
			if board[x][y] == '*' {
				board[x][y] = 'O'
			} else if board[x][y] == 'O' {
				board[x][y] = 'X'
			}
		}
	}
}

func sbfs(x, y int, board [][]byte) {
	// 종료 조건에 *이 있어서 중복 방문을 막아줌.
	if x < 0 || y < 0 || x >= len(board) || y >= len(board[x]) || board[x][y] == 'X' || board[x][y] == '*' {
		return
	}

	board[x][y] = '*'
	sbfs(x+1, y, board)
	sbfs(x-1, y, board)
	sbfs(x, y+1, board)
	sbfs(x, y-1, board)
}
