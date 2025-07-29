
def print_result_as_md(results, width=10):
    
    for key, value in results.items():
        first_row = "| item        | "
        item_number = len(value) + 1
        for k, v in value.items():
            first_row += f"{k:<{width}} |"
        break

    print(first_row)
    print("|" + "----------|" * item_number)

    for key, value in results.items():
        row = f"| {key:<{width}} | "
        for k, v in value.items():
            row += f"{v:<{width}.3f} |"
        print(row)


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import argparse

def send_email_body(smtp_server, smtp_port, sender_email, sender_password, receiver_email, subject, body):
    # 构造邮件
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # 连接 SMTP 服务器并发送邮件
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

def send_email():
    parser = argparse.ArgumentParser(description="Send an email via SMTP from the command line")
    parser.add_argument('--smtp_server', required=True, help='SMTP server address (e.g., smtp.gmail.com)')
    parser.add_argument('--smtp_port', type=int, default=465, help='SMTP server port (default: 465)')
    parser.add_argument('--sender_email', required=True, help='Sender email address')
    parser.add_argument('--sender_password', required=True, help='Sender email password or app token')
    parser.add_argument('--receiver_email', required=True, help='Receiver email address')
    parser.add_argument('--subject', required=True, help='Email subject')
    parser.add_argument('--body', required=True, help='Email body text')

    args = parser.parse_args()

    send_email_body(
        args.smtp_server,
        args.smtp_port,
        args.sender_email,
        args.sender_password,
        args.receiver_email,
        args.subject,
        args.body
    )

if __name__ == "__main__":
    send_email()

    