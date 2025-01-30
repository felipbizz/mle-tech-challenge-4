#!/usr/bin/env python
import pika, sys, os


def get_single_message():
    credentials = pika.PlainCredentials('rabbitmq', 'rabbitmq')
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost',credentials=credentials))
    channel = connection.channel()
    
    method_frame, header_frame, body = channel.basic_get('tasks')
    if method_frame:
        print(method_frame, header_frame, body)
        channel.basic_ack(method_frame.delivery_tag)
    else:
        print('No message returned')

def main():
    credentials = pika.PlainCredentials('rabbitmq', 'rabbitmq')
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost',credentials=credentials))
    channel = connection.channel()

    channel.queue_declare(queue='tasks')

    def callback(ch, method, properties, body):
        print(f" [x] Received {body}")

    channel.basic_consume(queue='tasks', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    # channel.start_consuming()
    channel.consume('tasks',auto_ack=False,exclusive=True)

    channel.close()

    connection.close()

if __name__ == '__main__':

    get_single_message()

    # try:
    #     main()
    # except KeyboardInterrupt:
    #     print('Interrupted')
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)