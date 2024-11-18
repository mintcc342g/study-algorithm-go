package linkedlists

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLinkedListReadAndIndexOf(t *testing.T) {
	type testCase struct {
		scenario string

		linkedList func(idx int, data string) *LinkedList

		expectedIdx int
		expected    string
	}
	testCases := []testCase{
		{
			scenario: "read from the end",
			linkedList: func(idx int, data string) *LinkedList {
				ll := NewLinkedList("dummy1")
				ll.InsertAt(1, "dummy2")
				ll.InsertAt(idx, data)

				return ll
			},
			expectedIdx: 2,
			expected:    "end",
		},
		{
			scenario: "read from the middle position",
			linkedList: func(idx int, data string) *LinkedList {
				ll := NewLinkedList("dummy1")
				ll.InsertAt(1, "dummy2")
				ll.InsertAt(idx, data)
				ll.InsertAt(3, "dummy3")

				return ll
			},
			expectedIdx: 2,
			expected:    "middle",
		},
		{
			scenario: "read from the beginning",
			linkedList: func(idx int, data string) *LinkedList {
				ll := NewLinkedList(data)
				ll.InsertAt(1, "dummy1")
				ll.InsertAt(2, "dummy2")
				ll.InsertAt(3, "dummy3")

				return ll
			},
			expectedIdx: 0,
			expected:    "beginning",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.scenario, func(t *testing.T) {
			ll := tc.linkedList(tc.expectedIdx, tc.expected)
			assert.Equal(t, tc.expected, ll.Read(tc.expectedIdx))
			assert.Equal(t, tc.expectedIdx, ll.IndexOf(tc.expected))
		})
	}
}

func TestLinkedListInsertAt(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"insert at the end": func(t *testing.T) {
			var (
				data     = "end"
				expected = 3
			)

			ll := NewLinkedList("dummy1")
			ll.InsertAt(1, "dummy2")
			ll.InsertAt(2, "dummy3")
			ll.InsertAt(3, "dummy4")
			ll.InsertAt(expected, data)

			assert.Equal(t, expected, ll.IndexOf(data))
		},
		"insert at the middle position": func(t *testing.T) {
			var (
				data     = "middle"
				expected = 2
			)

			ll := NewLinkedList("dummy1")
			ll.InsertAt(1, "dummy2")
			ll.InsertAt(2, "dummy3")
			ll.InsertAt(expected, data)

			assert.Equal(t, expected, ll.IndexOf(data))
		},
		"insert at the beginning": func(t *testing.T) {
			var (
				data     = "beginning"
				expected = 0
			)

			ll := NewLinkedList("dummy1")
			ll.InsertAt(1, "dummy2")
			ll.InsertAt(2, "dummy3")
			ll.InsertAt(expected, data)

			assert.Equal(t, expected, ll.IndexOf(data))
		},
	}
	for scenario, fn := range testCases {
		t.Run(scenario, fn)
	}
}

func TestLinkedListDeleteAt(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"delete at the end": func(t *testing.T) {
			var (
				idx      = 3
				data     = "del"
				expected = -1
			)

			ll := NewLinkedList("dummy1")
			ll.InsertAt(1, "dummy2")
			ll.InsertAt(2, "dummy3")
			ll.InsertAt(idx, data)

			ll.DeleteAt(idx)

			assert.Equal(t, expected, ll.IndexOf(data))
		},
		"delete at the middle position": func(t *testing.T) {
			var (
				idx      = 1
				data     = "del"
				expected = "dummy2"
			)

			ll := NewLinkedList("dummy1")
			ll.InsertAt(idx, data)
			ll.InsertAt(2, expected)
			ll.InsertAt(3, "dummy3")

			ll.DeleteAt(idx)

			assert.Equal(t, expected, ll.Read(idx))
		},
		"delete at the beginning": func(t *testing.T) {
			var (
				idx      = 0
				data     = "del"
				expected = "dummy1"
			)

			ll := NewLinkedList(data)
			ll.InsertAt(1, expected)
			ll.InsertAt(2, "dummy2")
			ll.InsertAt(3, "dummy3")

			ll.DeleteAt(idx)

			assert.Equal(t, expected, ll.Read(idx))
		},
	}
	for scenario, fn := range testCases {
		t.Run(scenario, fn)
	}
}
