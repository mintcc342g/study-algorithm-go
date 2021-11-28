package stack

import (
	"fmt"

	"github.com/pkg/errors"
)

func NewFixedStack(capa int) *FixedStack {
	return &FixedStack{
		stk:      make([]int, capa),
		capacity: capa,
	}
}

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

// isEmpty ...
func (s *FixedStack) isEmpty() bool {
	return s.ptr <= 0
}

// isFull ...
func (s *FixedStack) isFull() bool {
	return s.ptr >= s.capacity
}

// Push ...
func (s *FixedStack) Push(value int) error {
	if s.isFull() {
		return errors.Errorf("capacity over")
	}

	s.stk[s.ptr] = value
	s.ptr++

	return nil
}

// Pop ...
func (s *FixedStack) Pop() (int, error) {
	if s.isEmpty() {
		return 0, errors.Errorf("empty stack")
	}

	s.ptr--

	return s.stk[s.ptr], nil
}

// Peek ...
func (s *FixedStack) Peek() (int, error) {
	if s.isEmpty() {
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
	if s.isEmpty() {
		return "empty stack"
	}

	return fmt.Sprintf("%v", s.stk[:s.ptr]) // 현재 스택의 마지막 값의 위치를 ptr 이 나타내기 때문에 ptr 을 넣어줌.
}

// linter example
type Linter struct {
	stack      []rune
	brace      map[rune]rune
	openBrace  map[rune]bool
	closeBrace map[rune]bool
}

func NewLinter() *Linter {
	return &Linter{
		stack: []rune{},
		brace: map[rune]rune{
			']': '[',
			'}': '{',
			')': '(',
		},
		openBrace: map[rune]bool{
			'[': true,
			'{': true,
			'(': true,
		},
		closeBrace: map[rune]bool{
			']': true,
			'}': true,
			')': true,
		},
	}
}

func (l *Linter) Lint(txt string) error {

	for _, r := range txt {
		if l.isOpeningBrace(r) {
			l.pushToStack(r)
		} else if l.isClosingBrace(r) {
			if l.closesMostRecentOpeningBrace(r) {
				l.popFromStack(r)
			} else {
				return errors.Errorf("incorrect closing brace[%s]", string(r))
			}
		}
	}

	if len(l.stack) > 0 {
		return errors.New("unclosed lint error")
	}

	return nil
}

func (l *Linter) pushToStack(r rune) {
	l.stack = append(l.stack, r)
}

func (l *Linter) popFromStack(r rune) {
	l.stack = l.stack[:len(l.stack)-1]
}

func (l *Linter) isOpeningBrace(r rune) bool {
	return l.openBrace[r]
}

func (l *Linter) isClosingBrace(r rune) bool {
	return l.closeBrace[r]
}

func (l *Linter) closesMostRecentOpeningBrace(cb rune) bool {
	return l.brace[cb] == l.stack[len(l.stack)-1]
}
