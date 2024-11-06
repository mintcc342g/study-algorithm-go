package set

import "errors"

type IntSet struct {
	elements []int  // 요소 저장
	index    []bool // 중복 방지
}

func (r *IntSet) Add(value int) error {
	if value < 0 || r.Contains(value) {
		return errors.New("invalid value or already exist")
	}

	r.expand(value)
	r.elements = append(r.elements, value)
	r.index[value] = true

	return nil
}

func (r *IntSet) expand(value int) {
	if value < len(r.index) {
		return
	}

	size := value + 1
	index := make([]bool, size)
	copy(index, r.index)
	r.index = index
}

func (r *IntSet) Remove(value int) error {
	if value < 0 || !r.Contains(value) {
		return errors.New("invalid value or not exist")
	}

	r.index[value] = false

	for i, v := range r.elements {
		if v == value {
			r.elements = append(r.elements[:i], r.elements[i+1:]...)
			return nil
		}
	}

	return errors.New("[FATAL] fail to remove the value in elements; should not be reach here")
}

func (r *IntSet) Contains(value int) bool {
	if value < 0 || value >= len(r.index) {
		return false
	}

	return r.index[value]
}
