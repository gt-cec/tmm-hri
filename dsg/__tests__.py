# tests.py: several unit tests of the DSG

import dsg, node

# create the DSG object
def test_init():
    scene = dsg.DSG(same_location_threshold = 4)
    return scene

# initialize with three dummy objects
def test_add_objects(scene) -> None:
    # test scene is a table with two books
    table = {
        "class": "table",
        "x": 10,
        "y": 5,
        "z": .5
    }
    book_left = {
        "class": "book",
        "x": 7,
        "y": 6,
        "z": 1.0
    }
    book_right = {
        "class": "book",
        "x": 12,
        "y": 7,
        "z": 1.0
    }
    scene.add_object(object_dict = table)
    scene.add_object(object_dict = book_left)
    scene.add_object(object_dict = book_right)
    return

# left book, slightly different location
def test_naive_book_left(scene) -> None:
    seen_objects = [
        {
            "class": "book",
            "x": 7.5,
            "y": 5.5,
            "z": 1.4
        }
    ]
    scene.update(seen_objects)
    return

# right book, moved quite a bit away
def test_motion_book_right(scene) -> None:
    seen_objects = [
        {
            "class": "book",
            "x": 18,
            "y": 8,
            "z": 1.2
        }
    ]
    scene.update(seen_objects)
    return

# both books are in roughly the same locations
def test_naive_books_both(scene) -> None:
    seen_objects = [
        # the right book has moved a little
        {
            "class": "book",
            "x": 18.5,
            "y": 8.5,
            "z": 1.2
        },
        # the left book has moved a little
        {
            "class": "book",
            "x": 7,
            "y": 5.3,
            "z": 1.1
        }
    ]
    scene.update(seen_objects)
    return

# both books have moved a lot
def test_motion_books_both(scene) -> None:
    seen_objects = [
        # the right book has moved a lot
        {
            "class": "book",
            "x": 24.5,
            "y": 13.5,
            "z": 1.2
        },
        # the left book has moved a lot
        {
            "class": "book",
            "x": 11,
            "y": 8.3,
            "z": 1.1
        }
    ]
    scene.update(seen_objects)
    return

# run through the tests
if __name__ == "__main__":
    # create the scene
    scene = test_init()

    # add objects
    test_add_objects(scene)
    assert scene.count() == 3, "Adding three objects should result in a graph with three objects, instead have " + str(scene.count())

    book_left = "O2"
    book_right = "O3"

    # naive case: see book left but slightly shifted
    test_naive_book_left(scene)
    assert scene.objects[book_left]["y"] == 5.5 and scene.objects[book_right]["y"] == 7, "Seeing Book Left at a slightly different location (naive case) did not shift the book's known location"

    # single object move case: see book right but way out there
    test_motion_book_right(scene)
    assert scene.objects[book_left]["x"] == 7.5 and scene.objects[book_right]["x"] == 18, "Seeing Book Right way further to the right (one-object moving case) did not move the book's known location"

    # multi object naive case: see book right and book left both to the left
    test_naive_books_both(scene)
    assert scene.objects[book_left]["x"] == 7 and scene.objects[book_right]["x"] == 18.5, "Seeing both books at a slightly different location (naive case) did not shift the books correctly."

    # multi object naive case: see book right and book left both to the left
    test_motion_books_both(scene)
    assert scene.objects[book_left]["x"] == 11 and scene.objects[book_right]["x"] == 24.5, "Seeing both books at a way different location (motion case) did not shift the books correctly."

    print("All tests passed.")
