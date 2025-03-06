package revnote

import "math"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 모든 노드값 비교해서 그 차이가 가장 작은 값을 구하는 문제.
// inorder traversal 를 해줘야함.
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

// 오름차순 정렬된 배열을 높이 균형 맞춘 트리로 만들어야 함.
// 이진탐색처럼 배열을 반으로 갈라서 중간값을 찾아 넣는 방식으로 해야함.
func sortedArrayToBST(nums []int) *TreeNode {
	return sbst(0, len(nums)-1, nums)
}

func sbst(start, end int, nums []int) *TreeNode {
	// 이진탐색처럼 작동하니까 이진탐색과 비슷한 종료조건
	if start > end {
		return nil
	}

	mid := start + (end-start)/2
	node := new(TreeNode)
	node.Val = nums[mid]
	node.Left = sbst(start, mid-1, nums)
	node.Right = sbst(mid+1, end, nums)

	return node
}
