import gradio as gr
from datetime import datetime


def format_transcript(history):
    """Преобразует список сообщений в простой текст переписки."""
    if not history:
        return "Переписка пока пуста."

    lines = []
    for i, msg in enumerate(history, start=1):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        if role == "user":
            speaker = "Пользователь"
        elif role == "assistant":
            speaker = "Ассистент"
        else:
            speaker = "Система"

        lines.append(f"{i}. {speaker}: {content}")

    return "\n".join(lines)


def generate_answer(user_message, history, rag_context):
    """
    Заглушка под RAG-логику.
    Позже сюда можно подставить:
    1) поиск по векторной базе,
    2) сбор контекста,
    3) вызов LLM,
    4) запись ответа в историю.
    """
    history = history or []

    # Добавляем сообщение пользователя
    history = history + [{"role": "user", "content": user_message}]

    # Заглушка ответа ассистента
    timestamp = datetime.now().strftime("%H:%M:%S")
    if rag_context.strip():
        answer = (
            f"Понял запрос. Сейчас у меня есть черновой контекст для RAG.\n\n"
            f"Время: {timestamp}\n"
            f"Запрос: {user_message}\n"
            f"Контекст: {rag_context[:300]}"
        )
    else:
        answer = (
            f"Это базовый фронтенд для RAG-диалога.\n"
            f"Время: {timestamp}\n"
            f"Ваш запрос: {user_message}\n\n"
            f"Здесь позже будет ответ модели."
        )

    # Добавляем сообщение ассистента
    history = history + [{"role": "assistant", "content": answer}]

    transcript_text = format_transcript(history)

    # Очистаем поле ввода, обновляем историю, чат и текстовую расшифровку
    return "", history, history, transcript_text


def clear_all():
    return "", [], [], "Переписка пока пуста."


with gr.Blocks(title="RAG Chat Frontend") as demo:
    gr.Markdown(
        """
        # RAG Chat Frontend
        Базовый интерфейс для диалогов, истории переписки и будущего подключения поиска по базе знаний.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Диалоги",
                height=520,
                placeholder="Здесь появится диалог...",
                autoscroll=True,
            )

            user_input = gr.Textbox(
                label="Сообщение",
                placeholder="Введите вопрос и нажмите Enter или кнопку Отправить",
                lines=3,
            )

            with gr.Row():
                send_btn = gr.Button("Отправить", variant="primary")
                clear_btn = gr.Button("Очистить")

        with gr.Column(scale=1):
            transcript = gr.Textbox(
                label="Текст переписки",
                value="Переписка пока пуста.",
                lines=28,
                interactive=False,
            )

            rag_context = gr.Textbox(
                label="RAG-контекст (заглушка)",
                placeholder="Сюда позже можно подставлять найденные фрагменты из базы знаний",
                lines=8,
            )

    chat_state = gr.State([])

    # Отправка по Enter
    user_input.submit(
        fn=generate_answer,
        inputs=[user_input, chat_state, rag_context],
        outputs=[user_input, chat_state, chatbot, transcript],
    )

    # Отправка по кнопке
    send_btn.click(
        fn=generate_answer,
        inputs=[user_input, chat_state, rag_context],
        outputs=[user_input, chat_state, chatbot, transcript],
    )

    # Очистка
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[user_input, chat_state, chatbot, transcript],
    )

demo.launch()   