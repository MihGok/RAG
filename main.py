import loading_workflow as workflow


TARGET_TOPIC = "Python программирование"
BATCH_SIZE = 10
DOWNLOAD_THRESHOLD = 7 

def main():
    # 1. Поиск и сбор данных (Stepik)
    loader, raw_courses = workflow.fetch_stepik_courses(
        topic=TARGET_TOPIC, 
        limit=40
    )
    
    if not raw_courses:
        return

    # 2. Интеллектуальный анализ (Local LLM)
    analyzed_results = workflow.analyze_courses_relevance(
        raw_courses=raw_courses,
        topic=TARGET_TOPIC,
        llm_endpoint=LLM_ENDPOINT,
        batch_size=BATCH_SIZE
    )


    workflow.print_top_results(analyzed_results, top_n=20)


    workflow.download_top_courses(
        loader=loader,
        analyzed_courses=analyzed_results,
        raw_courses=raw_courses,
        min_score=DOWNLOAD_THRESHOLD,

        topic=TARGET_TOPIC,
        llm_endpoint=LLM_ENDPOINT
    )

if __name__ == "__main__":
    main()