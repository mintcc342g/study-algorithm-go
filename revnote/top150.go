package revnote

import "math"

/*
 **
 * 53. Maximum Subarray
 * 하위 배열의 합 중에서 가장 큰 값을 찾는 문제
 * 카데인 알고리즘
 * max 값이랑 sum 값을 별도로 저장하면서 찾음.
 */
func maxSubArray(nums []int) int {
	max, sum := nums[0], nums[0]
	for _, n := range nums[1:] {
		// 배열엔 마이너스 값도 들어있으므로, sum은 미리 더해줌.
		sum += n

		// 이때까지의 합보다 n이 더 크다면 n부터 새로 더하는 게 더 나음.
		if sum < n {
			sum = n
		}

		// 이때까지의 합과 max 비교해서 max 갱신
		if max < sum {
			max = sum
		}
	}

	return max
}

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

/*
 **
 * 215. Kth Largest Element in an Array
 * k번째로 큰 수 찾기
 * 효율이 가장 좋은 heap sort를 사용해서 해결
 * 효율이 나쁘지만 가능한 방법은 quick selection 정도인듯?
 */
func findKthLargest(nums []int, k int) int {
	// k번째로 큰 수 찾기
	// 배열이 들어오므로, heap sort로 우선순위큐로 만들면서 정렬
	// 그리고 k번째까지 root를 마지막 leaf로 바꾸면서 재구성하면 됨.

	l := len(nums) - 1
	for i := l / 2; i > 0; i-- { // 배열의 절반뒤는 항상 리프노드니까 앞 절반을 root로 넣어서 heap 구성
		heapify(nums, i, 0, l)
	}

	ret := 0
	for k > 0 {
		ret = nums[0]
		nums[0], nums[l] = nums[l], nums[0]
		l--
		heapify(nums, 0, 0, l)
		k--
	}

	return ret
}

func heapify(nums []int, parent, min, max int) {
	left := parent*2 + 1  // 공식
	right := parent*2 + 2 // 공식
	largest := parent

	if left >= min && left <= max && nums[left] > nums[largest] {
		largest = left
	}
	if right >= min && right <= max && nums[right] > nums[largest] {
		largest = right
	}

	if parent != largest {
		nums[parent], nums[largest] = nums[largest], nums[parent]
		heapify(nums, largest, min, max)
	}
}

/*
 **
 * 373. Find K Pairs with Smallest Sums
 * 주어진 배열 2개에서 숫자 하나씩 뽑아서 만든 합이 최소인 k개의 쌍을 리턴하는 문제
 * 배열은 오름차순으로 들어옴.
 */
func kSmallestPairs(nums1 []int, nums2 []int, k int) [][]int {
	// k번째 작은 수 등은 heap sort 가 유리한데, 여기서는 두 배열의 합을 만들어야 함.
	// 리턴해야 하는 건 두 배열의 각 인덱스이므로, 인덱스도 갖고 있어야 됨.
	// 따라서 두 배열의 합을 노드값으로 삼는 min heap 만들어야 함.

	// 단, 이러면 두 배열의 모든 조합을 만들어야 하므로 효율적이지 않음.
	// 따라서, nums1의 모든 수와 nums2의 0번째 값만으로 k개까지만 힙을 만들어줌.
	// (오름차순으로 오니까 어쨌든 둘 중 한 배열의 0번째 값만 쓰는 게 값이 작을 거라서.)

	// 그리고 k번째까지 쭉 돌면서 최소값 인덱스 뽑아내고, 값 하나씩 뽑아낼 때마다 힙을 재정렬함.
	// 이 때 어차피 힙을 재정렬해야 하므로, nums2의 수들을 뽑으면서 노드에 넣으면서 재정렬 해줌.

	// 구현 방법
	// 1. min heap 자료구조를 구현함. (upheap, downheap 구현 필요)
	// 2. min heap으로 k개까지 힙을 만들어줌.
	// 3. k번째까지 거꾸로 돌면서 1) 힙에서 노드를 하나씩 뽑고, 2) nums2[*] 로 힙을 다시 구성해줌.
	//    - 거꾸로 도는 이유는 변수 더 선언하기 싫어서인듯. 0부터 시작하려면 또다른 변수 선언해줘야 하니까
	heap := NewMinHeap()
	for i := 0; i < len(nums1) && i < k; i++ {
		heap.Push(Pair{sum: nums1[i] + nums2[0], i: i, j: 0})
	}

	ret := make([][]int, 0)
	for k > 0 {
		p := heap.Pop()
		ret = append(ret, []int{nums1[p.i], nums2[p.j]})

		if p.j+1 < len(nums2) {
			heap.Push(Pair{sum: nums1[p.i] + nums2[p.j+1], i: p.i, j: p.j + 1})
		}

		k--
	}

	return ret
}

type Pair struct {
	sum int
	i   int
	j   int
}

type MinHeap struct {
	pairs []Pair
}

func NewMinHeap() *MinHeap {
	return &MinHeap{pairs: []Pair{}}
}

// 받은 값을 힙의 가장 마지막 위치에 넣고 root로 올라오면서 자리 찾아줌.
func (r *MinHeap) Push(p Pair) {
	r.pairs = append(r.pairs, p)
	r.upHeap(len(r.pairs) - 1)
}

func (r *MinHeap) upHeap(child int) {
	parent := (child - 1) / 2 // 부모 찾는 공식
	if parent >= 0 && r.pairs[parent].sum > r.pairs[child].sum {
		r.pairs[parent], r.pairs[child] = r.pairs[child], r.pairs[parent]
		r.upHeap(parent)
	}
}

// 데이터 삭제 시에 사용
// 루트를 복제하고, 마지막 자식 노드를 루트로 올리고, 마지막 자식 노드를 삭제한 후에,
// 루트에 있는 값이 자리를 root -> leaf 순으로 찾아가게 해줌.
func (r *MinHeap) Pop() Pair {
	last := len(r.pairs) - 1
	if last < 0 {
		panic("empty heap")
	}

	root := r.pairs[0]
	r.pairs[0], r.pairs[last] = r.pairs[last], r.pairs[0]
	r.pairs = r.pairs[:last]
	r.downHeap(0)

	return root
}

// 입력받은 parent 위치부터 root -> leaf 순으로 내려가면서 정렬을 해줌.
func (r *MinHeap) downHeap(parent int) {
	left := parent*2 + 1  // 0-based 공식
	right := parent*2 + 2 // 0-based 공식
	smallest := parent

	if 0 <= left && left < len(r.pairs) && r.pairs[parent].sum > r.pairs[left].sum {
		smallest = left
	}
	if 0 <= right && right < len(r.pairs) && r.pairs[parent].sum > r.pairs[right].sum {
		smallest = right
	}
	if smallest != parent {
		r.pairs[smallest], r.pairs[parent] = r.pairs[parent], r.pairs[smallest]
		r.downHeap(smallest)
	}
}

/*
 **
 * 136. Single Number
 * 배열에서 2번 안 나온 수 찾아서 리턴하기
 * 단, linear 시간 복잡도, 상수 공간 복잡도 안에 해결해야 함.
 * 즉 map 쓰면 안 되고, 다른 방법을 찾아야 함.
 */
func singleNumber(nums []int) int {
	// 비트 연산 중 ^ 를 써야함.
	// 1개 수에 ^ 이거 쓰면(단항 연산자), 1은 0으로 0은 1로 바꿈. (NOT)
	// 근데 2개 수를 ^ 로 계산할 경우 둘을 합치는데(이항 연산자),
	// 수가 같으면 0이고, 수가 다르면 1로 바꿈. (XOR)
	// 예를 들어, 0100 + 1000 이라면 1100 이 된다는 뜻
	// 이걸 int로 보자면, 1 ^ 1 값은 0이 됨.
	// 근데 0 ^ 3 이라면 값은 3이 됨.
	// 따라서, nums 중 1개 수만 2번이 안 나오므로,
	// 걍 for문 돌리면서 이항 연산을 해주면 됨.

	res := 0
	for _, n := range nums {
		res ^= n
	}

	return res
}

/*
 **
 * 198. House Robber
 * 입력받은 배열에서 연속되지 않는 배열의 값을 더했을 때 나올 수 있는 최대값을 구하는 문제
 */
func rob(nums []int) int {
	// 시작할 때 0번째 인덱스부터 더할 것인지 1번째 인덱스부터 더할 것인지는 둘의 값을 비교해서 정함.
	// 현재 인덱스의 값을 더할지 말지를 전전인덱스의 값을 더해서 나온 값이 최대값이냐 아니냐로 결정함.
	// 이 떄문에 반드시 한 칸은 건너뛰게 되어 있고, 계속해서 최대값을 구하기 때문에
	// 결과적으로 최대값이 나오게 되어 있음.

	n := len(nums)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return nums[0]
	}

	pprev, prev := 0, nums[0] // 이게 무조건 초기값
	for _, num := range nums[1:] {
		pprev, prev = prev, max(prev, pprev+num)
	}

	return prev
}

func max(a, b int) int {
	if a > b {
		return a
	}

	return b
}

/*
 **
 * 198. House Robber
 * 입력받은 배열에서 연속되지 않는 배열의 값을 더했을 때 나올 수 있는 최대값을 구하는 문제
 * 단, 0번째 배열을 최대값을 구하는 데에 사용했을 경우 마지막 인덱스의 값은 사용해서는 안 됨.
 */
func rob2(nums []int) int {
	// 주어진 배열에서 인접하지 않은 값들의 최대값을 구하는 문제
	// 이건 0~마지막-1 또는 1~마지막까지의 합 중 큰 걸 리턴해야 함.
	// 0번째 사용했다고 마지막 안 더하는 식으로 if문 두면 예외에 걸려서 문제가 안 풀림.
	l := len(nums)
	if l == 0 {
		return 0
	}
	if l == 1 {
		return nums[0]
	}

	return max(robb(nums[:l-1]), robb(nums[1:]))
}

func robb(nums []int) int {
	pprev, prev := 0, nums[0]
	for _, num := range nums[1:] {
		pprev, prev = prev, max(prev, pprev+num)
	}
	return prev
}
