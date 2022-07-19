from magniv.core import task
import os
import json
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To


@task(
    schedule="@weekly", description="Run model weekly to send new book recommendations"
)
def email_me_new_books():
    # Pick three random books from my list of favorite books
    my_book_list = json.load(open("./tasks/datas/my_book_list.json"))
    selected_books = random.sample(my_book_list, 3)
    recommended_books = _get_recommended_books(selected_books)
    _send_email(recommended_books, selected_books)


def _send_email(recommended_books, selected_books):
    recommended_books_string = "<br>- ".join(recommended_books)
    selected_books_string = "<br>- ".join(selected_books)
    email_content = f"Hi<br><br>You new book recs are<br>- {recommended_books_string}<br>______<br>Taken from:<br>- {selected_books_string}<br>Enjoy"
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


def _get_recommended_books(current_books):
    book_to_index = json.load(open("./tasks/datas/book_to_index.json"))
    index_to_book = json.load(open("./tasks/datas/index_to_book.json"))
    v_matrix = np.load("./tasks/datas/v_matrix.npy")
    book_names = list(book_to_index.keys())
    nbrs = NearestNeighbors(n_neighbors=8, algorithm="ball_tree").fit(v_matrix)
    selected_indexes = [book_to_index[book] for book in current_books]
    vectors = []
    for index in selected_indexes:
        vectors.append(v_matrix[int(index), :])
    recommended_books = []
    if len(vectors) > 0:
        centroid = np.dstack(vectors).squeeze().mean(axis=1).transpose().reshape(-1, 1)
        closest_neighbors = nbrs.kneighbors(centroid.transpose(), return_distance=False)
        for n in closest_neighbors.flatten():
            recommended_books.append(index_to_book[str(n)])
    return recommended_books


if __name__ == "__main__":
    email_me_new_books()
