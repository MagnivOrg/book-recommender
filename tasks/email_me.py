from magniv.core import task
import os
import json
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To


def _send_email(reccomended_books, selected_books):
    email_content = f"Hi<br><br>You new book recs are<br>- {reccomended_books}<br>______<br>Taken from:<br>- {selected_books}<br>Enjoy"
    message = Mail(
        os.environ.get("FROM_EMAIL"),
        to_emails=[To(email=os.environ.get("TO_EMAIL"))],
        subject="Your Weekly Book Reccomendations",
        html_content=email_content,
    )

    try:
        sg = SendGridAPIClient(os.environ.get("SENDGRID_API_KEY"))
        response = sg.send(message)
    except Exception as e:
        print("pass ", e)


def _get_reccomended_books(current_books):
    book_to_index = json.load(open("./datas/book_to_index.json"))
    index_to_book = json.load(open("./datas/index_to_book.json"))
    v_matrix = np.load("./datas/v_matrix.npy")
    book_names = list(book_to_index.keys())
    nbrs = NearestNeighbors(n_neighbors=8, algorithm="ball_tree").fit(v_matrix)
    selected_indexes = [book_to_index[book] for book in current_books]
    vectors = []
    for index in selected_indexes:
        vectors.append(v_matrix[int(index), :])
    reccomended_books = []
    if len(vectors) > 0:
        centroid = np.dstack(vectors).squeeze().mean(axis=1).transpose().reshape(-1, 1)
        closest_neighbors = nbrs.kneighbors(centroid.transpose(), return_distance=False)
        for n in closest_neighbors.flatten():
            reccomended_books.append(index_to_book[str(n)])
    return reccomended_books


@task(schedule="@weekly")
def email_me_books():
    # Pick three random books from my list of faviorite books
    my_book_list = [
        "The Outer Limits of Reason: What Science, Mathematics, and Logic Cannot Tell Us",
        "Complexity: The Emerging Science at the Edge of Order and Chaos",
        "Reinventing the Sacred: A New View of Science, Reason, and Religion",
        "The Emperor's New Mind Concerning Computers, Minds and the Laws of Physics",
        "Shadows of the Mind: A Search for the Missing Science of Consciousness",
        "Antifragile: Things That Gain from Disorder",
        "The Enigma of Reason",
        "Thinking, Fast and Slow",
        "Good Natured - The Origins of Right & Wrong in Humans & Other Animals",
        "East of Eden",
        "The Age of Innocence",
        "Life of Pi",
        "Man Is Not Alone: A Philosophy of Religion",
        "Of Mice and Men",
        "The Four Agreements: A Practical Guide to Personal Freedom",
        "The Mastery of Love: A Practical Guide to the Art of Relationship --Toltec Wisdom Book",
        "The Art of Loving",
    ]
    selected_books = random.sample(my_book_list, 3)
    reccomended_books = _get_reccomended_books(selected_books)
    _send_email("<br>- ".join(reccomended_books), "<br>- ".join(selected_books))


if __name__ == "__main__":
    email_me_books()
