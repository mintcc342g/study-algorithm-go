package queue

import (
	"github.com/pkg/errors"
)

// 일반적인 배열로 만드는 큐는 복잡도가 O(n) 인데, 선입선출을 지키려고 배열 내 원소 이동이 필요하기 때문
// 링 버퍼 큐는 배열 내 원소를 옮길 필요 없이 front와 rear 의 값만 업데이트 하여 큐를 구현하므로 복잡도가 O(1)
// 링 버퍼는 오래된 데이터는 버리는 용도로 사용할 수 있음.

type FixedQueue struct {
	num      int   // 현재 데이터 개수
	front    int   // 맨 앞 원소의 커서
	rear     int   // 맨 뒤 원소의 커서
	capacity int   // 큐의 크기
	que      []int // 큐의 데이터
}

func NewFixedQueue(capa int) *FixedQueue {
	return &FixedQueue{
		capacity: capa,
		que:      make([]int, capa),
	}
}

func (q *FixedQueue) Len() int {
	return q.num
}

func (q *FixedQueue) isEmpty() bool {
	return q.num <= 0
}

func (q *FixedQueue) isFull() bool {
	return q.num >= q.capacity
}

func (q *FixedQueue) EnQueue(value int) error {
	if q.isFull() {
		return errors.Errorf("capacity over")
	}

	q.que[q.rear] = value // 마지막 자리에 값 넣어줌.
	q.rear++
	q.num++

	if q.rear == q.capacity { // 링 버퍼이기 때문에 최대 용량이 되면 rear 도 0번 인덱스로 보내는 것
		q.rear = 0
	}

	return nil
}

func (q *FixedQueue) DeQueue() (int, error) {
	if q.isEmpty() {
		return 0, errors.Errorf("empty queue")
	}

	result := q.que[q.front]
	q.front++
	q.num--

	if q.front == q.capacity { // 얘는 배열 인덱스의 한계를 넘어가는 걸 막기 위해서임.
		q.front = 0
	}

	return result, nil
}

func (q *FixedQueue) Peek() (int, error) {
	if q.isEmpty() {
		return 0, errors.Errorf("empty queue")
	}

	return q.que[q.front+1], nil
}

func (q *FixedQueue) Find(value int) (int, error) { // 찾는 값이 큐의 몇번째 인덱스에 들어있는지 반환
	if q.isEmpty() {
		return 0, errors.Errorf("empty queue")
	}

	var idx int
	for i := 0; i < q.num; i++ { // 선형검색을 하는데, queue 의 크기와 실제 값이 들어있는 양이 다르기 때문에 num 으로 for문 돌림.
		idx = (i + q.front) % q.capacity // 값이 들어있지 않은 곳은 제외하고, front 부터 검색하기 위한 계산식. % capa 를 해야 배열 인덱스를 초과시키지 않고 검색이 가능해짐.
		if q.que[idx] == value {
			return idx, nil
		}
	}

	return 0, errors.Errorf("not found")
}

func (q *FixedQueue) Count(value int) (int, error) {
	if q.isEmpty() {
		return 0, errors.Errorf("empty queue")
	}

	var cnt int
	var idx int
	for i := 0; i < q.num; i++ { // 여기도 Find 랑 같은 이유임.
		idx = (i + q.front) % q.capacity
		if q.que[idx] == value {
			cnt++
		}
	}

	if cnt == 0 {
		return 0, errors.Errorf("not found")
	}

	return cnt, nil
}

func (q *FixedQueue) Contains(value int) bool {
	if _, err := q.Count(value); err == nil {
		return true
	}

	return false
}

func (q *FixedQueue) Clear() {
	q.num, q.front, q.rear = 0, 0, 0
}

func (q *FixedQueue) Dump() { // 큐에 있는 모든 데이터를 맨 앞부터 맨 끝 순으로 출력
	if q.isEmpty() {
		return
	}

	var idx int
	for i := 0; i < q.num; i++ {
		idx = (i + q.front) % q.capacity
		println("dump", "idx:", idx, "value:", q.que[idx])
	}
}

type PrintManager struct {
	queue []string
}

func NewPrintManager() *PrintManager {
	return &PrintManager{
		queue: []string{},
	}
}

func (p *PrintManager) addJob(job string) {
	p.queue = append(p.queue, job)
}

func (p *PrintManager) run() {
	for range p.queue {
		job := p.shift()
		println(job)
	}
}

func (p *PrintManager) shift() (r string) {
	r = p.queue[0]
	if len(p.queue) != 0 {
		p.queue = p.queue[1:]
	}
	return
}
