"""
front.py
─────────
RAG CourseBuilder — Gradio UI в стиле Gemini.

Вкладки (скрыты за логином):
  Sidebar : список чатов + режим (Course / RAG)
  Main    : чат с AI, скачивание DOCX

Режимы:
  📚 Создать курс  — многошаговый диалог → Stage1 + Stage2 → DOCX
  🔍 Быстрый ответ — RAG-поиск по существующей базе знаний

Запуск: python front.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Generator, Optional

import gradio as gr

import loading_workflow as wf

EMBED_MODEL  = "Qwen3-Embedding-0.6B-f16.gguf"
MAIN_MODEL   = wf.MAIN_MODEL
from dotenv import load_dotenv
import os

load_dotenv()
# ─── DB (lazy) ───────────────────────────────────────────────────────────────

def _pg():
    try:
        import db.postgres_service as pg
        pg.init_db()
        return pg
    except Exception as e:
        print(f"!!! ОШИБКА ПОДКЛЮЧЕНИЯ К БД: {e}") # Выведет конкретную ошибку
        import traceback
        traceback.print_exc() 
        return None


def _qdrant_search():
    try:
        from db.qdrant_indexer import search_similar
        return search_similar
    except Exception:
        return None


# ─── AUTH ────────────────────────────────────────────────────────────────────

def do_login(email: str, password: str) -> tuple:
    pg = _pg()
    if not pg:
        return (
            gr.update(visible=True),   # login_block
            gr.update(visible=False),  # app_block
            "База данных недоступна.",
            {},
        )
    try:
        user = pg.authenticate_user(email.strip(), password)
        if not user:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                "Неверный email или пароль.",
                {},
            )
        user_state = {"user_id": user["user_id"], "email": user["email"],
                      "full_name": user.get("full_name", "")}
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            user_state,
        )
    except Exception as e:
        return gr.update(visible=True), gr.update(visible=False), f"❌ {e}", {}


def do_register(email: str, password: str, full_name: str) -> tuple:
    pg = _pg()
    if not pg:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "⚠️ База данных недоступна.",
            {},
        )
    if not email.strip() or not password.strip():
        return gr.update(visible=True), gr.update(visible=False), "❌ Заполните email и пароль.", {}
    try:
        user = pg.create_user(email.strip(), password, full_name.strip())
        user_state = {"user_id": user["user_id"], "email": user["email"],
                      "full_name": user.get("full_name", "")}
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            user_state,
        )
    except Exception as e:
        return gr.update(visible=True), gr.update(visible=False), f"❌ {e} (возможно, email уже занят)", {}


def do_logout(user_state: dict) -> tuple:
    return (
        gr.update(visible=True),   # login_block
        gr.update(visible=False),  # app_block
        "",
        {},
        [],
        _empty_conv(),
        [],
        gr.update(visible=False),
        None,
    )


# ─── CONV STATE ──────────────────────────────────────────────────────────────

def _empty_conv() -> dict:
    return {
        "stage":           "idle",    # idle | questions | generating | done
        "topic":           "",
        "questions":       [],
        "user_answers":    "",
        "chat_id":         None,
        "session_dir":     "",
        "collection_name": "",
        "docx_path":       "",
    }


# ─── CHAT LIST ───────────────────────────────────────────────────────────────

def _load_chat_list(user_state: dict) -> list[tuple[str, str]]:
    pg = _pg()
    if not pg or not user_state.get("user_id"):
        return []
    try:
        chats = pg.list_chats(user_state["user_id"])
        return [(f"{c['title'] or 'Без названия'} #{c['chat_id']}", str(c["chat_id"]))
                for c in chats]
    except Exception:
        return []


def _chat_choices(user_state: dict) -> list[str]:
    return [label for label, _ in _load_chat_list(user_state)]


def refresh_chats(user_state: dict) -> gr.update:
    return gr.update(choices=_chat_choices(user_state))


def select_chat(choice: str, user_state: dict) -> tuple:
    """Загружает сообщения выбранного чата."""
    pg = _pg()
    if not pg or not choice:
        return [], _empty_conv(), gr.update(visible=False), None

    # Извлекаем chat_id из строки вида "Название #123"
    try:
        chat_id = int(choice.rsplit("#", 1)[-1].strip())
    except (ValueError, IndexError):
        return [], _empty_conv(), gr.update(visible=False), None

    try:
        msgs = pg.get_messages(chat_id, limit=300)
        history = [
            {"role": "user" if m["sender_role"] else "assistant",
             "content": m["content"]}
            for m in msgs
        ]

        # Пытаемся восстановить meta из чата
        chats = pg.list_chats(user_state.get("user_id", 0))
        meta  = {}
        for c in chats:
            if c["chat_id"] == chat_id:
                raw = c.get("meta") or {}
                meta = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                break

        conv = _empty_conv()
        conv["chat_id"]         = chat_id
        conv["stage"]           = meta.get("stage", "done")
        conv["collection_name"] = meta.get("collection_name", "")
        conv["docx_path"]       = meta.get("docx_path", "")

        # Показываем кнопку скачивания если есть docx
        docx = conv["docx_path"]
        if docx and os.path.exists(docx):
            return history, conv, gr.update(visible=True), docx
        return history, conv, gr.update(visible=False), None

    except Exception as e:
        return [], _empty_conv(), gr.update(visible=False), None


def new_chat(user_state: dict) -> tuple:
    return (
        [],              # chatbot
        _empty_conv(),   # conv_state
        gr.update(visible=False),  # download_col
        None,            # course_file
        None,            # chat_selector value
    )


# ─── DB HELPERS ──────────────────────────────────────────────────────────────

def _ensure_chat(conv: dict, user_state: dict, title: str) -> int:
    pg = _pg()
    if not pg:
        return 0
    chat_id = conv.get("chat_id")
    if not chat_id:
        row = pg.create_chat(user_state.get("user_id", 0), title=title[:60])
        chat_id = row["chat_id"]
        conv["chat_id"] = chat_id
    return chat_id


def _save_pair(conv: dict, user_state: dict, user_msg: str, ai_msg: str):
    pg = _pg()
    if not pg:
        return
    try:
        chat_id = _ensure_chat(conv, user_state, user_msg)
        pg.add_message(chat_id, True,  user_msg)
        pg.add_message(chat_id, False, ai_msg)
    except Exception as e:
        print(f"[DB] {e}")


def _persist_meta(conv: dict):
    pg = _pg()
    if not pg or not conv.get("chat_id"):
        return
    try:
        pg.update_chat_meta(conv["chat_id"], {
            "stage":           conv.get("stage", "idle"),
            "collection_name": conv.get("collection_name", ""),
            "docx_path":       conv.get("docx_path", ""),
        })
    except Exception:
        pass


# ─── RAG QUICK ANSWER ────────────────────────────────────────────────────────

def _rag_answer(question: str, conv: dict) -> str:
    collection = conv.get("collection_name", "")
    if not collection:
        return ("⚠️ Нет активной базы знаний. "
                "Создайте курс в режиме «📚 Создать курс» сначала.")

    search_fn = _qdrant_search()
    if not search_fn:
        return "⚠️ Qdrant недоступен."

    try:
        hits = search_fn(question, collection, EMBED_MODEL, limit=3)
        if not hits:
            return "По вашему вопросу ничего не найдено в базе знаний."

        context = "\n\n".join(
            f"[{h.get('final_title','')}]\n{h.get('text','')[:500]}"
            for h in hits
        )
    except Exception as e:
        return f"⚠️ Ошибка поиска: {e}"

    import requests
    from config import AppConfig
    url = f"{AppConfig.ML_SERVER_URL.rstrip('/')}/task"

    prompt = (
        f"Контекст из базы знаний:\n\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        f"Дай краткий точный ответ, опираясь только на контекст."
    )
    try:
        resp = requests.post(url, json={
            "task_type":   "llm",
            "model_name":  MAIN_MODEL,
            "text":        prompt,
            "max_tokens":  512,
            "temperature": 0.3,
            "n_ctx":       4096,
        }, timeout=90)
        resp.raise_for_status()
        result = resp.json().get("result", {})
        return result.get("response", str(result)) or "Не удалось сгенерировать ответ."
    except Exception as e:
        return f"⚠️ Ошибка LLM: {e}"


# ─── COURSE PIPELINE (background + streaming) ────────────────────────────────

def _run_pipeline_bg(
    topic: str,
    user_answers: str,
    clarifying_questions: list,
    conv: dict,
    user_state: dict,
    log_buffer: list,
    result_holder: dict,
    done_event: threading.Event,
    transcribe: bool,
):

    try:
        def _log(msg: str):
            log_buffer.append(msg)
 
        # ── Stage 1 ──────────────────────────────────────────────────────
        session_dir = wf.run_stage1(
            topic=topic,
            user_answers=user_answers,
            clarifying_questions=clarifying_questions,   # ← НОВОЕ
            max_courses=5,
            limit_per_query=30,
            transcribe=transcribe,
            log_fn=_log,
        )
 
        if not session_dir:
            result_holder["error"] = "Stage 1 не вернул папку сессии"
            return
 
        # ── Stage 2 ──────────────────────────────────────────────────────
        from KnowledgeBaseCreator.pipeline import run_stage2
 
        chat_id = conv.get("chat_id")
        result  = run_stage2(
            session_dir=session_dir,
            user_id=user_state.get("user_id", 0),
            chat_id=chat_id,
            log_fn=_log,
        )
        result["session_dir"] = session_dir
        result_holder["result"] = result
 
    except Exception as e:
        result_holder["error"] = str(e)
        log_buffer.append(f"❌ Критическая ошибка: {e}")
    finally:
        done_event.set()

# ─── MESSAGE HANDLER ─────────────────────────────────────────────────────────

def handle_message(
    message: str,
    history: list,
    conv: dict,
    user_state: dict,
    mode: str,
    transcribe: bool,
) -> Generator:
    """
    Центральный обработчик сообщений пользователя.
    Yields (history, conv, download_col_update, file_value, chat_selector_update).
    """
    message = (message or "").strip()
    if not message:
        yield history, conv, gr.update(visible=False), None, gr.update()
        return

    # Добавляем сообщение пользователя
    history = list(history) + [{"role": "user", "content": message}]
    yield history, conv, gr.update(visible=False), None, gr.update()

    # ── Режим быстрого ответа ────────────────────────────────────────────
    if mode == "🔍 Быстрый ответ":
        answer = _rag_answer(message, conv)
        history = history + [{"role": "assistant", "content": answer}]
        _save_pair(conv, user_state, message, answer)
        yield history, conv, gr.update(visible=False), None, gr.update()
        return

    # ── Режим создания курса ─────────────────────────────────────────────
    stage = conv.get("stage", "idle")

    # Stage: idle → задаём уточняющие вопросы
    if stage == "idle":
        conv = {**conv, "topic": message, "stage": "questions"}
        questions = wf.generate_clarifying_questions(message)
        conv["questions"] = questions

        q_text = "\n\n".join(f"**{i+1}.** {q}" for i, q in enumerate(questions))
        ai_msg = (
            f"Отличная тема для курса! Чтобы создать максимально "
            f"подходящий учебный материал, ответьте на несколько вопросов:\n\n"
            f"{q_text}\n\n"
            f"*Можно ответить на все вопросы в одном сообщении.*"
        )
        history = history + [{"role": "assistant", "content": ai_msg}]
        _save_pair(conv, user_state, message, ai_msg)
        _ensure_chat(conv, user_state, message)
        yield history, conv, gr.update(visible=False), None, gr.update()
        return

    if stage == "questions":
        conv = {**conv, "user_answers": message, "stage": "generating"}
        topic = conv.get("topic", message)
 
        # Начальное сообщение
        history = history + [{"role": "assistant", "content": "🚀 Запускаю создание курса..."}]
        yield history, conv, gr.update(visible=False), None, gr.update()
 
        # Запускаем pipeline в фоне
        log_buffer:    list  = []
        result_holder: dict  = {}
        done_event = threading.Event()
 
        t = threading.Thread(
            target=_run_pipeline_bg,
            args=(
                topic,
                message,
                conv.get("questions", []),   # ← НОВОЕ: передаём уточняющие вопросы
                conv,
                user_state,
                log_buffer,
                result_holder,
                done_event,
                transcribe,
            ),
            daemon=True,
        )
        t.start()

        # Стримим прогресс
        last_len = 0
        while not done_event.is_set():
            time.sleep(1.5)
            if len(log_buffer) > last_len:
                last_len = len(log_buffer)
                progress = "\n".join(f"• {l}" for l in log_buffer[-10:])
                history[-1]["content"] = (
                    f"⚙️ **Создаю курс...**\n\n"
                    f"```\n{progress}\n```"
                )
                yield history, conv, gr.update(visible=False), None, gr.update()

        t.join()

        if "error" in result_holder:
            err = result_holder["error"]
            history[-1]["content"] = f"❌ Ошибка создания курса:\n\n{err}"
            conv = {**conv, "stage": "idle"}
            _save_pair(conv, user_state, message, history[-1]["content"])
            yield history, conv, gr.update(visible=False), None, gr.update()
            return

        result = result_holder.get("result", {})
        conv = {
            **conv,
            "stage":           "done",
            "session_dir":     result.get("session_dir", ""),
            "collection_name": result.get("collection_name", ""),
            "docx_path":       result.get("docx_path", ""),
        }
        _persist_meta(conv)

        docx_path     = result.get("docx_path", "")
        lessons_count = result.get("lessons_count", 0)
        chunks_count  = result.get("chunks_count", 0)
        modules_count = result.get("modules_count", 0)

        ai_msg = (
            f"✅ **Курс успешно создан!**\n\n"
            f"📊 **Статистика:**\n"
            f"- Уроков обработано: {lessons_count}\n"
            f"- Тематических блоков: {chunks_count}\n"
            f"- Модулей в курсе: {modules_count}\n\n"
            f"📥 **Документ готов к скачиванию** — нажмите кнопку ниже.\n\n"
            f"💡 *Переключитесь в режим «🔍 Быстрый ответ» для поиска "
            f"по созданной базе знаний.*"
        )
        history[-1]["content"] = ai_msg
        _save_pair(conv, user_state, message, ai_msg)

        # Обновляем список чатов
        new_choices = _chat_choices(user_state)

        if docx_path and os.path.exists(docx_path):
            yield (
                history, conv,
                gr.update(visible=True),
                docx_path,
                gr.update(choices=new_choices),
            )
        else:
            yield (
                history, conv,
                gr.update(visible=False),
                None,
                gr.update(choices=new_choices),
            )
        return

    # Stage: done → RAG по созданной базе
    if stage == "done":
        answer  = _rag_answer(message, conv)
        history = history + [{"role": "assistant", "content": answer}]
        _save_pair(conv, user_state, message, answer)
        yield history, conv, gr.update(visible=False), None, gr.update()


# ════════════════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Layout ── */
body { background: #f0f4f9 !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }

/* ── Login card ── */
#login-card {
    max-width: 420px;
    margin: 80px auto;
    background: #fff;
    border-radius: 16px;
    padding: 40px 36px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10);
}
#login-card h1 { font-size: 1.7em; font-weight: 700; color: #1a1a2e; margin-bottom: 6px; }
#login-card p  { color: #666; margin-bottom: 24px; font-size: 0.95em; }

/* ── App shell ── */
#app-shell { height: 100vh; display: flex; background: #f0f4f9; }

/* ── Sidebar ── */
#sidebar {
    width: 280px;
    min-width: 220px;
    background: #e8edf5;
    border-right: 1px solid #d0d7e3;
    display: flex;
    flex-direction: column;
    padding: 0;
    height: 100vh;
    overflow-y: auto;
}
#sidebar-logo {
    padding: 20px 20px 10px;
    font-size: 1.2em;
    font-weight: 700;
    color: #1a5fb4;
    letter-spacing: -0.3px;
}
#new-chat-btn {
    margin: 8px 12px;
    border-radius: 24px !important;
    background: transparent !important;
    border: 1px solid #b0bec5 !important;
    color: #333 !important;
    font-size: 0.92em !important;
    padding: 8px 14px !important;
}
#new-chat-btn:hover { background: #d0dde8 !important; }
.chat-section-label {
    padding: 12px 20px 4px;
    font-size: 0.75em;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
#chat-selector { padding: 0 8px; }
#chat-selector .wrap { gap: 2px !important; }
#chat-selector label {
    border-radius: 10px !important;
    padding: 9px 14px !important;
    font-size: 0.88em !important;
    cursor: pointer !important;
    color: #333 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
#chat-selector label:hover { background: #d8e3f0 !important; }
#chat-selector input[type=radio]:checked + label {
    background: #c8d8ee !important;
    font-weight: 600 !important;
}
#mode-selector { padding: 8px 12px; }
#mode-selector .wrap { gap: 4px !important; }
#mode-selector label {
    border-radius: 10px !important;
    padding: 7px 12px !important;
    font-size: 0.87em !important;
}
#transcribe-check { padding: 4px 14px 4px; }
.sidebar-divider { height: 1px; background: #c8d5e3; margin: 8px 16px; }
#logout-btn {
    margin: 8px 12px 16px;
    border-radius: 24px !important;
    background: transparent !important;
    border: 1px solid #c9b4b4 !important;
    color: #a33 !important;
    font-size: 0.88em !important;
}
#logout-btn:hover { background: #fde8e8 !important; }
#user-badge {
    padding: 10px 16px;
    font-size: 0.82em;
    color: #555;
    border-top: 1px solid #c8d5e3;
}

/* ── Main content ── */
#main-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #fff;
    overflow: hidden;
}
#chat-header {
    padding: 18px 28px 14px;
    border-bottom: 1px solid #e8ecf0;
    background: #fff;
    font-size: 1.1em;
    font-weight: 600;
    color: #1a1a2e;
}
#main-chatbot {
    flex: 1;
    overflow-y: auto;
}
#main-chatbot .message-wrap { padding: 12px 28px !important; }
#download-col {
    background: #f0f7ff;
    border-top: 1px solid #c8dff5;
    padding: 12px 24px;
}
#download-col h4 { margin: 0 0 8px; color: #1a5fb4; font-size: 0.95em; }
#input-row {
    padding: 14px 24px 18px;
    border-top: 1px solid #e8ecf0;
    background: #fff;
}
#msg-input textarea {
    border-radius: 24px !important;
    border: 1.5px solid #c8d5e3 !important;
    padding: 12px 18px !important;
    font-size: 0.95em !important;
    resize: none !important;
    background: #f7f9fc !important;
}
#msg-input textarea:focus {
    border-color: #1a5fb4 !important;
    background: #fff !important;
    box-shadow: 0 0 0 3px rgba(26,95,180,0.12) !important;
}
#send-btn {
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    font-size: 1.3em !important;
    padding: 0 !important;
    background: #1a5fb4 !important;
    color: #fff !important;
    border: none !important;
    margin-top: 4px;
}
#send-btn:hover { background: #154fa0 !important; }
"""

# ════════════════════════════════════════════════════════════════════════════
#  BUILD UI
# ════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="CourseRAG") as demo:

    # ── Persistent state ──────────────────────────────────────────────────
    user_state = gr.State({})
    conv_state = gr.State(_empty_conv())

    # ══════════════════════════════════════════════════════════════════════
    # LOGIN BLOCK
    # ══════════════════════════════════════════════════════════════════════
    with gr.Column(visible=True, elem_id="login-card") as login_block:
        gr.HTML("<h1>📚 CourseRAG</h1><p>Система генерации учебных курсов на основе AI</p>")

        with gr.Tab("Вход"):
            login_email    = gr.Textbox(label="Email", placeholder="you@example.com")
            login_password = gr.Textbox(label="Пароль", type="password",
                                        placeholder="••••••••")
            login_error    = gr.Markdown("")
            login_btn      = gr.Button("Войти", variant="primary")

        with gr.Tab("Регистрация"):
            reg_name     = gr.Textbox(label="Имя", placeholder="Иван Иванов")
            reg_email    = gr.Textbox(label="Email", placeholder="you@example.com")
            reg_password = gr.Textbox(label="Пароль (мин. 6 символов)", type="password")
            reg_error    = gr.Markdown("")
            reg_btn      = gr.Button("Создать аккаунт", variant="primary")

    # ══════════════════════════════════════════════════════════════════════
    # MAIN APP BLOCK
    # ══════════════════════════════════════════════════════════════════════
    with gr.Row(visible=False, elem_id="app-shell") as app_block:

        # ── Sidebar ──────────────────────────────────────────────────────
        with gr.Column(elem_id="sidebar", scale=0, min_width=260):
            gr.HTML('<div id="sidebar-logo">📚 CourseRAG</div>')

            new_chat_btn = gr.Button("+ Новый чат", elem_id="new-chat-btn")

            gr.HTML('<div class="chat-section-label">Чаты</div>')
            chat_selector = gr.Radio(
                choices=[],
                label="",
                elem_id="chat-selector",
                interactive=True,
            )

            gr.HTML('<div class="sidebar-divider"></div>')
            gr.HTML('<div class="chat-section-label">Режим</div>')
            mode_radio = gr.Radio(
                choices=["📚 Создать курс", "🔍 Быстрый ответ"],
                value="📚 Создать курс",
                label="",
                elem_id="mode-selector",
            )

            transcribe_check = gr.Checkbox(
                value=False,
                label="Транскрибировать видео",
                elem_id="transcribe-check",
                info="Снимите для ускорения без видео",
            )

            gr.HTML('<div class="sidebar-divider"></div>')

            user_badge   = gr.HTML('<div id="user-badge">—</div>')
            logout_btn   = gr.Button("Выйти", elem_id="logout-btn")

        # ── Main content ─────────────────────────────────────────────────
        with gr.Column(elem_id="main-col", scale=1):
            chat_header = gr.HTML(
                '<div id="chat-header">Начните новый чат →</div>'
            )

            chatbot = gr.Chatbot(
                value=[],
                height=None,
                elem_id="main-chatbot",
                
                placeholder=(
                    "**Добро пожаловать в CourseRAG!**\n\n"
                    "Введите тему — и система создаст для вас полноценный учебный курс:\n"
                    "- Поиск материалов на Stepik\n"
                    "- Кластеризация и структурирование знаний\n"
                    "- Генерация DOCX-документа\n\n"
                    "В режиме «🔍 Быстрый ответ» задавайте вопросы по уже созданной базе."
                ),
                show_label=False,
                buttons= ['copy'    ]
            )

            # Download section (скрыт по умолчанию)
            with gr.Column(visible=False, elem_id="download-col") as download_col:
                gr.HTML("<h4>📥 Курс готов к скачиванию</h4>")
                course_file = gr.File(
                    label="Скачать курс (DOCX)",
                    interactive=False,
                )

            # Input row
            with gr.Row(elem_id="input-row"):
                msg_input = gr.Textbox(
                    show_label=False,
                    placeholder="Введите тему курса или задайте вопрос...",
                    lines=1,
                    max_lines=6,
                    scale=10,
                    elem_id="msg-input",
                )
                send_btn = gr.Button("➤", scale=0, elem_id="send-btn", variant="primary")

    # ══════════════════════════════════════════════════════════════════════
    # EVENT WIRING
    # ══════════════════════════════════════════════════════════════════════

    _auth_outputs = [login_block, app_block, login_error, user_state]

    def _post_login(user_state_val):
        """После логина: обновляем список чатов и бейдж."""
        choices = _chat_choices(user_state_val)
        email   = user_state_val.get("email", "")
        name    = user_state_val.get("full_name", "") or email
        badge   = f'<div id="user-badge">👤 {name}</div>'
        return gr.update(choices=choices), badge

    # Login
    login_btn.click(
        do_login,
        inputs=[login_email, login_password],
        outputs=_auth_outputs,
    ).then(
        _post_login,
        inputs=[user_state],
        outputs=[chat_selector, user_badge],
    )
    login_password.submit(
        do_login,
        inputs=[login_email, login_password],
        outputs=_auth_outputs,
    ).then(
        _post_login,
        inputs=[user_state],
        outputs=[chat_selector, user_badge],
    )

    # Register
    reg_btn.click(
        do_register,
        inputs=[reg_email, reg_password, reg_name],
        outputs=[login_block, app_block, reg_error, user_state],
    ).then(
        _post_login,
        inputs=[user_state],
        outputs=[chat_selector, user_badge],
    )

    # Logout
    logout_btn.click(
        do_logout,
        inputs=[user_state],
        outputs=[
            login_block, app_block, login_error, user_state,
            chat_selector, conv_state, chatbot, download_col, course_file,
        ],
    )

    # New chat
    new_chat_btn.click(
        new_chat,
        inputs=[user_state],
        outputs=[chatbot, conv_state, download_col, course_file, chat_selector],
    ).then(
        lambda: gr.update(value='<div id="chat-header">Новый чат</div>'),
        outputs=[chat_header],
    )

    # Select existing chat
    def _on_select_chat(choice, user_state_val):
        hist, conv, dl_upd, fval = select_chat(choice, user_state_val)
        topic = conv.get("topic", "") or "Чат"
        header = f'<div id="chat-header">{topic}</div>'
        return hist, conv, dl_upd, fval, header

    chat_selector.change(
        _on_select_chat,
        inputs=[chat_selector, user_state],
        outputs=[chatbot, conv_state, download_col, course_file, chat_header],
    )

    # Send message
    _send_outputs = [
        chatbot, conv_state, download_col, course_file, chat_selector
    ]

    def _send(msg, hist, conv, user_st, mode, transcribe):
        # update header when topic is set for the first time
        for state in handle_message(msg, hist, conv, user_st, mode, transcribe):
            yield state + (gr.update(),)  # last = cleared input (handled separately)

    send_btn.click(
        handle_message,
        inputs=[msg_input, chatbot, conv_state, user_state, mode_radio, transcribe_check],
        outputs=_send_outputs,
    ).then(
        lambda: gr.update(value=""),
        outputs=[msg_input],
    )
    msg_input.submit(
        handle_message,
        inputs=[msg_input, chatbot, conv_state, user_state, mode_radio, transcribe_check],
        outputs=_send_outputs,
    ).then(
        lambda: gr.update(value=""),
        outputs=[msg_input],
    )


# ─── LAUNCH ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )