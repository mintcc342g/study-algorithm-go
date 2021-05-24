package stack

import (
	"fmt"

	"github.com/pkg/errors"
)

// FixedStack ...
type FixedStack struct {
	stk      []int // stack
	capacity int
	ptr      int // pointer
}

// Len ...
func (s *FixedStack) Len() int {
	return s.ptr
}

// IsEmpty ...
func (s *FixedStack) IsEmpty() bool {
	return s.ptr <= 0
}

// IsFull ...
func (s *FixedStack) IsFull() bool {
	return s.ptr >= s.capacity
}

// Push ...
func (s *FixedStack) Push(value int) error {
	if s.IsFull() {
		return errors.Errorf("capacity over")
	}

	s.stk[s.ptr] = value
	s.ptr++

	return nil
}

// Pop ...
func (s *FixedStack) Pop() (int, error) {
	if s.IsEmpty() {
		return 0, errors.Errorf("empty stack")
	}

	s.ptr--

	return s.stk[s.ptr], nil
}

// Peek ...
func (s *FixedStack) Peek() (int, error) {
	if s.IsEmpty() {
		return 0, errors.Errorf("empty stack")
	}

	return s.stk[s.ptr-1], nil
}

// Clear ...
func (s *FixedStack) Clear() {
	s.ptr = 0
}

// Find ...
func (s *FixedStack) Find(value int) (int, error) {
	for i := s.ptr - 1; i >= 0; i-- { // 뒤부터 찾는 이유는 먼저 Pop 할 수 있는 값을 찾기 위해서라는 듯
		if s.stk[i] == value {
			return i, nil
		}
	}

	return 0, errors.Errorf("not found")
}

// Count ...
func (s *FixedStack) Count(value int) int {
	var count int

	for _, v := range s.stk {
		if v == value {
			count++
		}
	}

	return count
}

// Contains ...
func (s *FixedStack) Contains(value int) bool {
	if _, err := s.Find(value); err != nil { // 책에서는 Count 를 사용했는데, 이유는 모르겠음.
		return false
	}

	return true
}

// Dump ...
func (s *FixedStack) Dump() string {
	if s.IsEmpty() {
		return fmt.Sprint("empty stack")
	}

	return fmt.Sprintf("%v", s.stk[:s.ptr]) // 현재 스택의 마지막 값의 위치를 ptr 이 나타내기 때문에 ptr 을 넣어줌.
}
