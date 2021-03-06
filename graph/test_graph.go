package graph

import "fmt"

func TestGraph() (err error) {

	println("\n///// Graph")

	riri := NewPerson("Riri")
	yuyu := NewPerson("Yuyu")
	mai := NewPerson("Mai")

	sachie := NewPerson("sachie")
	raimu := NewPerson("raimu")
	clara := NewPerson("clara")

	riri.AddFriends(
		NewPerson("Kaede"),
		NewPerson("Fumi"),
		NewPerson("Miliam"),
		NewPerson("Tazusa"),
		NewPerson("Yujia"),
		NewPerson("Shenrin"),
		yuyu,
		mai,
	)

	yuyu.AddFriends(
		sachie,
	)

	mai.AddFriends(
		NewPerson("soraha"),
	)

	sachie.AddFriends(
		raimu,
	)

	raimu.AddFriends(
		NewPerson("seren"),
		clara,
	)

	clara.AddFriends(
		NewPerson("himari"),
	)

	println(fmt.Sprintf("\n// start display %s's network by dfs", riri.name))
	riri.DFS()

	println(fmt.Sprintf("\n// start display %s's network by bfs", riri.name))
	riri.BFS()

	return nil
}
