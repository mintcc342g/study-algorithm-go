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
			if h.elements[left] < h.elements[right] {
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
