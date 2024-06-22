import tkinter as tk
from tkinter import messagebox, Listbox, Button, Entry, Label, Toplevel, Scrollbar
from tkhtmlview import HTMLLabel
from imapclient import IMAPClient
import email
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup

# Загрузка и обучение модели наивного байеса
data = pd.read_csv('spam.csv')
data['label'] = data['category'].map({'spam': 1, 'ham': 0})
texts = data['text'].values
labels = data['label'].values
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

def is_spam(content):
    # Преобразование текста так же, как и для обучения модели
    content_cleaned = re.sub(r'\s+', ' ', content).strip()
    content_vector = vectorizer.transform([content_cleaned])
    prediction = model.predict(content_vector)[0]
    return prediction == 1

def move_spam_to_folder(mail, msg_id):
    try:
        mail.copy([msg_id], 'Спам')  # Измените на 'спам' или 'Junk', в зависимости от структуры папок
        mail.delete_messages([msg_id])
        mail.expunge()
        #messagebox.showinfo("Успех", f"Письмо {msg_id} перемещено в папку спам.")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось переместить письмо в папку спам.\nОшибка: {str(e)}")

def fetch_folders(mail):
    try:
        folders = mail.list_folders()
        return [folder[2] for folder in folders]
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось получить список папок.\nОшибка: {str(e)}")
        return []

def decode_mime_words(s):
    decoded_words = []
    for word, encoding in email.header.decode_header(s):
        if encoding:
            decoded_words.append(word.decode(encoding))
        else:
            decoded_words.append(word if isinstance(word, str) else word.decode('utf-8', 'ignore'))
    return ''.join(decoded_words)

def fetch_emails(mail, folder):
    try:
        mail.select_folder(folder)
        messages_listbox.delete(0, tk.END)
        messages = mail.search()
        for msg_id in messages:
            msg = mail.fetch([msg_id], ['ENVELOPE'])[msg_id]
            envelope = msg[b'ENVELOPE']
            msg_subject = decode_mime_words(envelope.subject.decode() if envelope.subject else '(Без темы)')
            msg_from = decode_mime_words(envelope.from_[0].name.decode() if envelope.from_[0].name else envelope.from_[0].mailbox.decode() + '@' + envelope.from_[0].host.decode())
            messages_listbox.insert(tk.END, f"От: {msg_from}, Тема: {msg_subject}, ID: {msg_id}")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось получить список писем.\nОшибка: {str(e)}")

def display_email(mail, msg_id):
    try:
        msg = mail.fetch([msg_id], ['RFC822'])[msg_id]
        email_message = email.message_from_bytes(msg[b'RFC822'])
        msg_subject = decode_mime_words(email_message['subject'])
        msg_from = decode_mime_words(email_message['from'])
        msg_body = ""

        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                if "attachment" not in content_disposition:
                    body_part = part.get_payload(decode=True)
                    if body_part:
                        msg_body += body_part.decode('utf-8', 'ignore')
        else:
            body_part = email_message.get_payload(decode=True)
            if body_part:
                msg_body = body_part.decode('utf-8', 'ignore')

        show_email_window(msg_subject, msg_from, msg_body)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось открыть письмо.\nОшибка: {str(e)}")

def show_email_window(subject, sender, body):
    email_window = Toplevel()
    email_window.title(f"Письмо: {subject}")

    lbl_subject = Label(email_window, text=f"Тема: {subject}")
    lbl_subject.pack()

    lbl_sender = Label(email_window, text=f"Отправитель: {sender}")
    lbl_sender.pack()

    html_label = HTMLLabel(email_window, html=body)
    html_label.pack(fill="both", expand=True)

def connect_to_mail():
    global mail
    try:
        mail = IMAPClient('imap.mail.ru', ssl=True)
        mail.login(entry_login.get(), entry_password.get())
        messagebox.showinfo("Успех", "Подключение к почтовому серверу выполнено успешно.")
        folders = fetch_folders(mail)
        for folder in folders:
            folder_listbox.insert(tk.END, folder)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось подключиться к серверу.\nОшибка: {str(e)}")



def clean_text(text):
    # Удаление HTML-тегов
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")

    # Удаление лишних пробелов и переводов строк
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text

def check_spam(mail):
    selected_folder = folder_listbox.get(tk.ACTIVE)
    if not selected_folder:
        messagebox.showwarning("Предупреждение", "Выберите папку для проверки на спам.")
        return

    try:
        mail.select_folder(selected_folder)
        messages_listbox.delete(0, tk.END)
        messages = mail.search()

        for msg_id in messages:
            raw_message = mail.fetch([msg_id], ['RFC822'])
            msg_data = raw_message[msg_id]
            msg = email.message_from_bytes(msg_data[b'RFC822'])

            msg_subject = msg.get('Subject', '')
            msg_from = msg.get('From', '')

            msg_body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))

                    if "attachment" not in content_disposition:
                        body_part = part.get_payload(decode=True)
                        if body_part:
                            text = body_part.decode('utf-8', 'ignore')
                            msg_body += clean_text(text) + "\n"
            else:
                body_part = msg.get_payload(decode=True)
                if body_part:
                    text = body_part.decode('utf-8', 'ignore')
                    msg_body = clean_text(text)

            if is_spam(msg_body):
                move_spam_to_folder(mail, msg_id)
            else:
                messages_listbox.insert(tk.END, f"От: {msg_from}, Тема: {msg_subject}, ID: {msg_id}")

    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось получить список писем.\nОшибка: {str(e)}")

root = tk.Tk()
root.title("Почтовый клиент")

label_login = Label(root, text="Логин:")
label_login.grid(row=0, column=0)
entry_login = Entry(root)
entry_login.grid(row=0, column=1)

label_password = Label(root, text="Пароль:")
label_password.grid(row=1, column=0)
entry_password = Entry(root, show='*')
entry_password.grid(row=1, column=1)

btn_connect = Button(root, text="Подключиться", command=connect_to_mail)
btn_connect.grid(row=2, columnspan=2)

folder_scrollbar = Scrollbar(root)
folder_scrollbar.grid(row=3, column=2, sticky=tk.N+tk.S)

folder_listbox = Listbox(root, yscrollcommand=folder_scrollbar.set, height=10, width=50)
folder_listbox.grid(row=3, column=0, columnspan=2)
folder_scrollbar.config(command=folder_listbox.yview)

btn_check_spam = Button(root, text="Проверить на спам", command=lambda: check_spam(mail))
btn_check_spam.grid(row=4, columnspan=2)

messages_scrollbar = Scrollbar(root)
messages_scrollbar.grid(row=5, column=2, sticky=tk.N+tk.S)

messages_listbox = Listbox(root, yscrollcommand=messages_scrollbar.set, height=10, width=100)
messages_listbox.grid(row=5, column=0, columnspan=2)
messages_scrollbar.config(command=messages_listbox.yview)

btn_open_email = Button(root, text="Открыть письмо", command=lambda: display_email(mail, int(messages_listbox.get(messages_listbox.curselection()).split(', ')[-1].split(': ')[-1])))
btn_open_email.grid(row=6, columnspan=2)

folder_listbox.bind("<<ListboxSelect>>", lambda e: fetch_emails(mail, folder_listbox.get(folder_listbox.curselection())))

root.mainloop()
