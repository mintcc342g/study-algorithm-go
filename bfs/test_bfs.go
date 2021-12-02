package bfs

import "fmt"

func TestBFS() (err error) {

	println("\n// start test bfs")

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

	println(fmt.Sprintf("start display %s's network", riri.name))
	riri.DisplayNetwork()

	return nil
}
