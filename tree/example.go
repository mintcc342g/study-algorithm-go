package tree

import "fmt"

// MinHeap:
// 부모 < 자식 이어야 하는 힙.
// MaxHeap은 이거 반대로만 해주면 됨.
type MinHeap struct {
	elements []uint32 // 얘가 사실상 우선순위 큐가 되는 것, 이건 1-based array임.
	size     uint32   // 힙에 원소를 넣을 수 있는 최대 사이즈
	last     uint32   // 원소 넣을 때 가장 마지막 위치부터 검색하면서 부모로 올라가기 위해서 필요함.
}

func NewMinHeap(size uint32) *MinHeap {
	size++ // 0-th 인덱스는 사용하지 않으니까 +1 해줌.

	return &MinHeap{
		elements: make([]uint32, size),
		size:     size,
		last:     0,
	}
}

func (h *MinHeap) Add(element uint32) {
	h.last++ // 하나 넣을 거니까 현재 last 늘려줌.

	// 최대 사이즈보다 많이 넣게 될 경우 안 넣고 끝냄.
	if h.last >= h.size {
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

// recursion 사용, upHeap이랑 downHeap내의 부등호만 바꾸면 MinHeap 됨.
type MaxHeap struct {
	elements []int32 // 이건 0-based array임.
}

func NewMaxHeap(isMin bool) *MaxHeap {
	return &MaxHeap{
		elements: make([]int32, 0),
	}
}

// 새로운 요소를 가장 마지막 자식 노드에 넣고,
// leaf -> root 방향으로 자리를 찾아감.
func (r *MaxHeap) Push(val int32) {
	r.elements = append(r.elements, val)
	r.upHeap(len(r.elements) - 1)
}

func (r *MaxHeap) upHeap(child int) {
	parent := (child - 1) / 2

	if parent >= 0 && r.elements[child] > r.elements[parent] {
		r.elements[child], r.elements[parent] = r.elements[parent], r.elements[child]
		r.upHeap(parent)
	}
}

// 현재 루트를 빼내고, 가장 마지막 자식을 루트로 올린 후에 heap 크기를 줄이고,
// root -> leaf 방향으로 자리를 찾아감.
func (r *MaxHeap) Pop() int32 {
	if len(r.elements) == 0 {
		panic("heap is empty!!")
	}

	root := r.elements[0]
	r.elements[0] = r.elements[len(r.elements)-1]
	r.elements = r.elements[:len(r.elements)-1]
	r.downHeap(0)

	return root
}

func (r *MaxHeap) downHeap(parent int) {
	left := parent*2 + 1
	right := parent*2 + 2
	largest := parent

	if left < len(r.elements) && r.elements[left] > r.elements[largest] {
		largest = left
	}
	if right < len(r.elements) && r.elements[right] > r.elements[largest] {
		largest = right
	}

	if largest != parent {
		r.elements[parent], r.elements[largest] = r.elements[largest], r.elements[parent]
		r.downHeap(largest)
	}
}

// leetcode: 트리를 가정한 nums 라는 배열에서 k번째로 큰 수 찾기
// (sorting 없이 min heap으로 해결해야 함)
func findKthLargest(nums []int, k int) int {

	// for문 절반만 도는 이유
	//   - nums가 tree 로만 온다는 조건이 있음.
	//   - 트리로 배열을 구성했을 경우, 배열의 뒤 절반은 리프 노드임.
	//   - 최대값을 찾는 문제이므로, min-heap 형태로 바꿔줘야 하는데,
	//   - 바꿀 때 부모 노드를 기준으로 잡아야 하니까 배열의 절반만 도는 거임.
	l := len(nums) - 1
	for i := l / 2; i > -1; i-- { // 주의할 점은 0부터 마지막 인덱스까지 전부 따진다는 것
		// 최대힙 형태로 만들어줌. 최대값 찾아야 하니까.
		heapify(nums, 0, l, i)
	}

	var res int

	// 이미 배열은 max heap 상태이며, k번째를 찾아야 함.
	// 밑의 방식은 현재 루트값과 자식값을 교환한 후에 heap을 재구성 하는 것
	// 루트값은 제외한 채로(lastIdx를 하나씩 줄여가니까 제외가 될 수 있음)
	// max-heap 을 다시 구성하는 건데,
	// 이렇게 하면, k번째까지 루트값이 삭제되면서 heap이 계속 만들어질 거고
	// 최종적으로 k번째가 루트로 올라오게 될거임.
	// 그 값이 res에 담길 거고.
	for i := 0; i < k; i++ {
		res = nums[0]
		nums[0], nums[l] = nums[l], nums[0]
		heapify(nums, 0, l-1, 0) // heap 재구성 하는데, 루트 마지막 인덱스 빼고, 무조건 0번 루트에서 시작함.
		l--
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

// Trie(GPT 버전)
type TrieNode struct {
	children map[rune]*TrieNode
	isEnd    bool
}

type Trie struct {
	root *TrieNode
}

// Trie 생성
func NewTrie() *Trie {
	return &Trie{root: &TrieNode{children: make(map[rune]*TrieNode)}}
}

// 삽입 (Insert)
func (t *Trie) Insert(word string) {
	node := t.root
	for _, ch := range word {
		if _, found := node.children[ch]; !found {
			node.children[ch] = &TrieNode{children: make(map[rune]*TrieNode)}
		}
		node = node.children[ch]
	}
	node.isEnd = true
}

// 검색 (Search)
func (t *Trie) Search(word string) bool {
	node := t.root
	for _, ch := range word {
		if _, found := node.children[ch]; !found {
			return false
		}
		node = node.children[ch]
	}
	return node.isEnd
}

// 접두어 검색 (StartsWith)
func (t *Trie) StartsWith(prefix string) bool {
	node := t.root
	for _, ch := range prefix {
		if _, found := node.children[ch]; !found {
			return false
		}
		node = node.children[ch]
	}
	return true
}

/*
 **
 * Tree: Huffman Decoding (해커랭크)
 * Huffman Tree란, 트리의 일종으로, 마지막 노드에만 문자가 있고
 * 그 문자에 도달하기 전까지의 노드는 문자를 갖지 않고 빈도수만 가짐.
 * 그리고 그 마지막 노드에 도달하기 까지의 경로를 0과 1로만 나타냄.
 */
func huffmanCoding() {
	//Enter your code here. Read input from STDIN. Print output to STDOUT
	var s string
	fmt.Scan(&s) // ABCAABARD

	// 문자열 내의 자소의 빈도수를 알아내기 위한 맵 구성
	m := map[rune]int{}
	for _, r := range s {
		m[r]++
	}

	// 일단 빈도수에 따라서 min heap을 만들어줌.
	// 이때는 아직 트리 간의 포인터가 연결 안 된 상태
	h := &MinHeapHuff{}
	for chara, freq := range m {
		h.Push(&NodeHuff{
			freq:  freq,
			chara: chara,
		})
	}

	// huffman tree 만들기
	for len(h.data) > 1 {
		// 1. 힙에서 2개 노드를 뺌
		left := h.Pop()
		right := h.Pop()

		// 2. 두 노드를 병합해줌
		// 이 때 트리 간에 연결을 해주는 것
		merged := &NodeHuff{
			freq:  left.freq + right.freq,
			left:  left,
			right: right,
		}

		// 3. 만든 노드를 기존 heap에 넣어줌
		// 이 과정을 반복하면 min heap 배열에는 최종적으로 root 노드만 남게 됨.
		h.Push(merged)
	}

	// 인코딩 시작
	root := h.Pop() // 힙에서 루트 노드 빼줌
	em := map[rune]string{}
	encodeHuff(root, "", em) // 말단 노드까지 가면서 코드 만들어줌. (내부 설명 참고)
	encoded := ""
	for _, r := range s {
		encoded += em[r] // 맵에 s의 각 자소에 대한 코드가 담겼을 거고, 그걸 다 합쳐서 인코딩 된 글자를 만들어줌. (e.g., 100101110)
	}
	fmt.Println(encoded)

	// 인코딩된 글자를 원본 글자로 바꿔주는 과정
	decoded := decodeHuff(root, encoded) // 여기서도 말단노드까지 가면서 글자를 만듦. (내부 설명 참고)
	fmt.Println(decoded)
}

func encodeHuff(node *NodeHuff, code string, em map[rune]string) {
	if node == nil {
		return
	}

	// 말단 노드에 도착했다면, 이때까지 축적된 인코딩된 코드를 맵에 넣어줌.
	if node.left == nil && node.right == nil {
		em[node.chara] = code // 말단 노드만 글자 갖고 있으니까, 그 글자까지 도달하는 데에 필요한 코드를 넣어줌
		return
	}

	// 말단 노드까지 가기
	// 왼쪽으로 갈땐 0을 붙여주고, 오른쪽으로 갈땐 1을 붙여줌
	encodeHuff(node.left, code+"0", em)
	encodeHuff(node.right, code+"1", em)
}

func decodeHuff(node *NodeHuff, encoded string) string {
	res := ""
	curr := node
	for _, code := range encoded {
		if code == '0' { // 인코딩 때와 마찬가지로, 0이면 왼쪽 1이면 오른쪽으로 감
			curr = curr.left
		} else {
			curr = curr.right
		}

		// 말단 노드에 도달했을 때 비로소 글자가 있을 거니까
		// 결과값에 글자를 붙여주고 다시 루트부터 다른 노드에 있는 글자를 찾으러 가야됨.
		if curr.left == nil && curr.right == nil {
			res += string(curr.chara)
			curr = node
		}
	}

	return res
}

type NodeHuff struct {
	freq  int
	chara rune
	left  *NodeHuff
	right *NodeHuff
}

type MinHeapHuff struct {
	data []*NodeHuff
}

// leaf -> root
func (r *MinHeapHuff) Push(node *NodeHuff) {
	r.data = append(r.data, node)
	r.upheap(len(r.data) - 1)
}

func (r *MinHeapHuff) upheap(child int) {
	parent := (child - 1) / 2
	if parent >= 0 && r.data[parent].freq > r.data[child].freq {
		r.data[parent], r.data[child] = r.data[child], r.data[parent]
		r.upheap(parent)
	}
}

// root -> leaf
func (r *MinHeapHuff) Pop() *NodeHuff {
	if len(r.data) == 0 {
		return nil
	}

	root := r.data[0]
	r.data[0] = r.data[len(r.data)-1]
	r.data = r.data[:len(r.data)-1]
	r.downheap(0)

	return root
}

func (r *MinHeapHuff) downheap(parent int) {
	left := parent*2 + 1
	right := parent*2 + 2
	smallest := parent

	if left < len(r.data) && r.data[left].freq < r.data[smallest].freq {
		smallest = left
	}
	if right < len(r.data) && r.data[right].freq < r.data[smallest].freq {
		smallest = right
	}
	if smallest != parent {
		r.data[smallest], r.data[parent] = r.data[parent], r.data[smallest]
		r.downheap(smallest)
	}
}
