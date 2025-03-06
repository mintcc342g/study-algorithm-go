package revnote

import "math"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type ListNode struct {
	Val  int
	Next *ListNode
}

/*
 **
 * 530. Minimum Absolute Difference in BST
 * 모든 노드값 비교해서 그 차이가 가장 작은 값을 구하는 문제.
 * inorder traversal 를 해줘야함.
 */
func getMinimumDifference(root *TreeNode) int {
	// 글로벌 변수 2개 필요(최소값, 이전 노드값)
	min := math.MaxInt64 // 비교를 시작해야 하므로, 발생할 수 있는 최대 차이값으로 초기화
	prev := -1           // 이전 노드에 없을법한 값으로 초기화

	// 2개 글로벌 변수 다루는 게 귀찮으니까 순회하는 함수도 걍 변수로 선언
	// inorder traversal은 왼쪽 -> 부모 -> 오른쪽 순서
	// 	    4
	// 	   / \
	//    2   6
	//   / \   \
	//  1   3   8
	// 이런 트리라면, 1->2->3->4->6->8 이 순서로 방문한다는 것
	var traverse func(node *TreeNode)
	traverse = func(node *TreeNode) {
		if node == nil { // 중료조건
			return
		}

		traverse(node.Left) // 왼쪽부터 순회

		// inorder traversal을 할 경우, 오름차순으로 정렬된 순서로 순회를 하게 됨.
		// 따라서, 항상 prev <= node.Val 이 됨.
		curr := node.Val - prev
		if prev != -1 && curr < min {
			min = curr // 조건에 따라 현재 노드를 비교한 최소값으로 갱신
		}
		prev = node.Val // 다음 노드를 가야하므로 이전 노드값을 갱신해줌.

		traverse(node.Right) // 오른쪽 순회
	}

	traverse(root) // 순회 시작

	return min
}

/*
 **
 * 108. Convert Sorted Array to Binary Search Tree
 * 오름차순 정렬된 배열을 높이 균형 맞춘 트리로 만들어야 함.
 * 이진탐색처럼 배열을 반으로 갈라서 중간값을 찾아 넣는 방식으로 해야함.
 */
func sortedArrayToBST(nums []int) *TreeNode {
	return sbst(0, len(nums)-1, nums)
}

func sbst(start, end int, nums []int) *TreeNode {
	// 이진탐색처럼 작동하니까 이진탐색과 비슷한 종료조건
	if start > end {
		return nil
	}

	mid := start + (end-start)/2 // 중간값 찾기
	node := new(TreeNode)
	node.Val = nums[mid]
	node.Left = sbst(start, mid-1, nums) // 왼쪽 절반
	node.Right = sbst(mid+1, end, nums)  // 오른쪽 절반

	return node
}

/*
 **
 * 109. Convert Sorted List to Binary Search Tree
 * linked list를 bst로 만드는 문제로, 여기선 2가지 방법이 가능함.
 * 첫 번째는 two pointers 로 중간값을 찾으면서 만들어가는 방식
 *   -> O(NlogN), 순환하는 배열도 가능
 * 두 번째는 linked list를 int 배열로 만든 후에 이진탐색처럼 중간값을 찾아서 만드는 방식
 *   -> O(N), 순환하는 linked list면 이 방법 x
 */
func sortedListToBST1(head *ListNode) *TreeNode {
	// two pointers 방식은 2칸씩 앞으로 가는 fast 포인터와 1칸씩 앞으로 가는 slow 포인터 2개를 이용하는 것
	// fast가 linked list의 끝에 도달했을 때, slow는 linked list의 중간 지점에 위치하게 됨.
	// 이 때 slow가 가리키는 노드의 값을 bst의 새로운 노드에 넣어주고, 리스트를 분할해줌.
	// 그리고 분할된 linked list의 앞부분과 뒷부분을 각각 재귀로 돌려줌.
	// 중요한 게 재귀 종료 조건인데, 단일 노드의 경우 무한 재귀가 발생하지 않도록 next가 nil인 경우를 꼭 걸러줘야함!

	if head == nil {
		return nil
	}
	// 이 조건문이 없으면 head가 단일 노드인 경우 무한 재귀가 발생
	// 왜냐면 하단에서 왼쪽 트리 만들 때 head값을 넣어주기 떄문
	if head.Next == nil {
		return &TreeNode{Val: head.Val}
	}

	prev, slow, fast := head, head, head
	// slow를 중간 위치까지 옮기기
	for fast != nil && fast.Next != nil {
		prev = slow
		slow = slow.Next
		fast = fast.Next.Next
	}

	// 순환하지 못하도록 중간 노드의 메모리 주소를 변수에서 없앰.
	// slow는 여전히 중간 노드부터의 메모리 주소를 갖고 있으므로 문제 x
	prev.Next = nil

	// 찾은 중간값을 bst에 새 노드 만들어서 넣어줌.
	res := &TreeNode{Val: slow.Val}
	res.Left = sortedListToBST1(head)       // 절반 앞 부분부터 왼쪽에 넣도록 시작점인 head를 넣어줌.
	res.Right = sortedListToBST1(slow.Next) // 절반 뒷 부분은 오른쪽 트리에 넣도록 slow.Next를 넣어줌.

	return res
}

func sortedListToBST2(head *ListNode) *TreeNode {
	// linked list를 배열로 만드는 방식은 이 문제에서 주어지는
	// linked list가 순환하지 않기 때문에 가능한 것임.
	// 우선 for문을 돌면서 linked list를 int 배열로 만들어줌.
	// 이후는 bst에서 했던 것처럼 중간값을 찾아서 넣어주면 됨.

	arr := []int{}
	for head != nil {
		arr = append(arr, head.Val)
		head = head.Next
	}

	return sbs(arr)
}

func sbs(arr []int) *TreeNode {
	if len(arr) == 0 {
		return nil
	}

	mid := len(arr) / 2
	node := new(TreeNode)
	node.Val = arr[mid]
	node.Left = sbs(arr[:mid])
	node.Right = sbs(arr[mid+1:])

	return node
}

/*
 **
 * 200. Number of Islands
 * '1'로 연결되어 있는 집합(섬)을 찾는 문제
 * bfs로 모든 셀 방문해서 값 변경하면서 카운팅
 */
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

/*
 **
 * 130. Surrounded Regions
 * 4면이 x로 둘러 쌓여있지 않은 o만 x로 바꿔주면 됨.
 * 이건 3면은 x로 둘러쌓였으면서 배열 가장 바깥에 위치한 o와
 * 걔랑 연결된 o는 바꾸면 안 된다는 뜻
 */
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
