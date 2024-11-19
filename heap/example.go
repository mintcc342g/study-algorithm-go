package heap

/**
 * MinHeap
 * 부모 < 자식 이어야 하는 힙
 * MaxHeap은 이거 반대로만 해주면 됨.
 */

type MinHeap struct {
	elements []uint32 // 얘가 사실상 우선순위 큐가 되는 것
	size     uint32   // 힙에 원소를 넣을 수 있는 최대 사이즈
	last     uint32   // 원소 넣을 때 가장 마지막 위치부터 검색하면서 부모로 올라가기 위해서 필요함.
}

func NewMinHeap(size uint32) *MinHeap {
	size++ // 힙은 0-th 인덱스는 사용하지 않으니까 +1 해줌.

	return &MinHeap{
		elements: make([]uint32, size),
		size:     size,
		last:     0,
	}
}

func (h *MinHeap) Add(element uint32) {
	h.last++ // 하나 넣을 거니까 현재 last 늘려줌.

	// 최대 사이즈보다 많이 넣게 될 경우 안 넣고 끝냄.
	if h.last > h.size {
		h.last-- // 원상복귀
		println("Added too many elements")
		return
	}

	h.elements[h.last] = element // 일단 힙의 맨 마지막 위치에 새 값 넣어줌.

	// 맞는 자리 찾기 시작
	childIdx := h.last
	parentIdx := childIdx / 2

	// 새로 넣은 값 < 부모 값 일 때, 자리 바꾸기 및 인덱스 재정의
	// 0번 인덱스는 안 쓰니까 childIdx 1보다 클 때를 조건으로 넣어줌.
	for h.elements[childIdx] < h.elements[parentIdx] && childIdx > 1 {
		h.elements[childIdx], h.elements[parentIdx] = h.elements[parentIdx], h.elements[childIdx]
		childIdx = parentIdx
		parentIdx = childIdx / 2
	}
}

// 가장 최상단 부모값을 뽑으면서 삭제함.
func (h *MinHeap) Pop() uint32 {
	if h.last < 1 {
		println("There is no element in the heap")
		return 0
	}

	var parentIdx uint32 = 1
	result := h.elements[parentIdx]
	h.elements[parentIdx] = h.elements[h.last] // 맨 마지막 자식을 루트로 올린 후, 다른 자식들과 비교하면서 자리 바꿔나감.
	h.last--

	// 삭제 시엔 완전 이진 트리가 아닌데다,
	// 새로운 루트값의 왼/오른쪽 중 큰 자식쪽만 타고 가서 한 쪽면만 비교를 하면 되니까 /2를 해주는 것
	for parentIdx <= h.last/2 {
		left := parentIdx * 2
		right := parentIdx*2 + 1

		if h.elements[parentIdx] > h.elements[left] || h.elements[parentIdx] > h.elements[right] {
			if h.elements[left] < h.elements[right] { // 우선순위 큐라서 왼쪽이 항상 오른쪽보다 작아야 함.
				h.elements[parentIdx], h.elements[left] = h.elements[left], h.elements[parentIdx]
				parentIdx = left
			} else {
				h.elements[parentIdx], h.elements[right] = h.elements[right], h.elements[parentIdx]
				parentIdx = right
			}
		} else {
			break
		}
	}

	return result
}

// leetcode: 트리를 가정한 nums 라는 배열에서 k번째로 큰 수 찾기
// (sorting 없이 min heap으로 해결해야 함)
func findKthLargest(nums []int, k int) int {

	// for문 절반만 도는 이유
	//   - nums가 tree 로만 온다는 조건이 있음.
	//   - 트리로 배열을 구성했을 경우, 배열의 뒤 절반은 리프 노드임.
	//   - 최대값을 찾는 문제이므로, min-heap 형태로 바꿔줘야 하는데,
	//   - 바꿀 때 부모 노드를 기준으로 잡아야 하니까 배열의 절반만 도는 거임.
	for i := len(nums)/2 - 1; i > -1; i-- { // 주의할 점은 0부터 마지막 인덱스까지 전부 따진다는 것
		// 최대힙 형태로 만들어줌. 최대값 찾아야 하니까.
		heapify(nums, 0, len(nums)-1, i)
	}

	var res int
	lastIdx := len(nums) - 1

	// 이미 배열은 max heap 상태이며, k번째를 찾아야 함.
	// 밑의 방식은 현재 루트값과 자식값을 교환한 후에 heap을 재구성 하는 것
	// 루트값은 제외한 채로(lastIdx를 하나씩 줄여가니까 제외가 될 수 있음)
	// max-heap 을 다시 구성하는 건데,
	// 이렇게 하면, k번째까지 루트값이 삭제되면서 heap이 계속 만들어질 거고
	// 최종적으로 k번째가 루트로 올라오게 될거임.
	// 그 값이 res에 담길 거고.
	for i := 0; i < k; i++ {
		res = nums[0]
		nums[0], nums[lastIdx] = nums[lastIdx], nums[0]
		heapify(nums, 0, lastIdx-1, 0) // heap 재구성 하는데, 루트 마지막 인덱스 빼고, 무조건 0번 루트에서 시작함.
		lastIdx--
	}

	return res
}

// 입력받은 parent를 기준으로, nums라는 트리를 max-heap 형태로 바꿔줌.
func heapify(nums []int, low int, high int, parent int) {
	// 자식 노드 인덱스 값은 공식임.
	left := 2*parent + 1
	right := 2*parent + 2

	larger := parent // 현재 부모노드와 자식 노드들 중 가장 큰 값을 갖는 자식 노드. 디폴트는 부모로 설정

	// 왼쪽 자식의 인덱스가 nums의 길이를 벗어나지 않으면서,
	if low <= left && left <= high {
		// 부모 값 < 왼쪽 값 이라면, 왼쪽을 부모로 올려줌.
		// 왜냐면 max heap은 부모 값이 더 커야해서.
		if nums[larger] < nums[left] {
			larger = left
		}
	}

	// 마찬가지로, 오른쪽 자식 인덱스가 nums의 길이를 벗어나지 않으면서,
	if low <= right && right <= high {
		// 부모 값 < 오른쪽 값 이라면, 오른쪽을 부모로 올려줌.
		if nums[larger] < nums[right] {
			larger = right
		}
	} // 둘 다 하는 이유는 자식 간의 순서를 정한다기 보다는, 부모가 두 자식 모두한테서 다 커야하기 때문임.

	// larger가 변경되었다면, 실제 배열에서의 위치도 바꿔줘야 함.
	if larger != parent {
		// 따라서, 현재 작은 수인 부모와 가장 큰 수인 자식(larger)의 배열 내 위치를 바꿔줌.
		nums[larger], nums[parent] = nums[parent], nums[larger]
		// 가장 큰 자식(larger) 위치에 이전 부모의 값이 들어갔음.
		// 따라서, 그 새로운 녀석의 자식들을 또 정렬 시키기 위해서 heapify를 재귀 호출함.
		heapify(nums, low, high, larger)
	}
}
