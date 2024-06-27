import pandas as pd
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings

model_db_pairs = [
        ('all-minilm:latest', 'minilm'),
        ('gte-base:latest', 'gte'),
        ('nomic-embed-text:latest', 'nomic'),
        ('qwen2:0.5b', 'qwen'),
        ('tinyllama:latest', 'tinyllama'),
        ('mxbai-embed-large:latest', 'mxbai'),
        ('gemma:2b-instruct-v1.1-q2_K', 'gemma'),
        ('phi3:mini-128k', 'phimini'),
        ('llama3:latest', 'llama'),
        ]
queries = [
        ("Sammanfatta de viktigaste fynden och rekommendationerna från forskningsartiklar om självmordsprevention med fokus på hur de kan tillämpas i ideella organisationers arbete.",
         "Summarize the key findings and recommendations from research articles on suicide prevention with a focus on how they can be applied to the work of non-profit organizations."),

        ("Summera nyckelinsikterna från studier om effektiviteten av olika självmordspreventiva metoder.",
         "Summarize the key insights from studies on the effectiveness of various suicide prevention methods."),

        ("Ge en översikt över forskningsresultaten som diskuterar nya trender inom självmordsprevention och hur dessa kan integreras i ideell sektors initiativ.",
         "Provide an overview of research findings that discuss emerging trends in suicide prevention and how these can be integrated into nonprofit sector initiatives."),

        ("Sammanfatta de huvudsakliga slutsatserna från artiklar som beskriver effektiva samtalstekniker vid krishantering och självmordsprevention.",
         "Summarize the main findings from articles describing effective conversational techniques in crisis management and suicide prevention."),

        ("Presentera översikter av forskning som belyser tecken på självmordsrisk och hur volontärer kan identifiera och agera i dessa samtal.",
         "Present overviews of research highlighting signs of suicidal risk and how volunteers can identify and act on these conversations."),

        ("Sammanfatta studiers huvudfynd om bästa praxis för att skapa förtroende och öppenhet under stödsamtal med personer som har självmordstankar.",
         "Summarize key study findings on best practices for building trust and openness during supportive conversations with people experiencing suicidal thoughts."),

        ("Summera forskningsresultat om effektiv utbildning och stöd till volontärer som arbetar med självmordsprevention.",
         "Summarize research findings on effective training and support for volunteers working in suicide prevention."),

        ("Ge en översikt över artiklar som diskuterar utmaningar och lösningar vid samordning av självmordspreventiva insatser bland volontärer.",
         "Provide an overview of articles that discuss challenges and solutions in coordinating suicide prevention efforts among volunteers."),

        ("Sammanfatta rekommendationer från forskning för att förbättra stöd och uppföljning av volontärer inom självmordsprevention.",
         "Summarize recommendations from research to improve support and follow-up of suicide prevention volunteers."),

        ("Sök efter artiklar som visar effektiva policy-åtgärder och kampanjer för att minska självmordsfrekvensen.",
         "Search for articles that demonstrate effective policy measures and campaigns to reduce suicide rates."),

        ("Identifiera forskningsartiklars huvudsakliga slutsatser om allmänhetens attityder till självmord och hur dessa kan förändras genom informationskampanjer.",
         "Identify the main conclusions of research articles about public attitudes towards suicide and how these can be changed through information campaigns."),

        ("Ge en översikt av artiklar som diskuterar sambandet mellan medierapportering och självmordsprevention och hur vi kan använda detta i påverkansarbete.",
         "Provide an overview of articles that discuss the relationship between media reporting and suicide prevention and how we can use this in advocacy work."),

        ("Sammanfatta viktiga insikter från forskning om organisatoriska faktorer som bidrar till framgångsrika självmordspreventiva insatser.",
         "Summarize important insights from research on organizational factors that contribute to successful suicide prevention efforts."),

        ("Summera forskningsartiklars huvudsakliga fynd om effektiva modeller för verksamhetsutveckling inom självmordspreventiva frivilligorganisationer.",
         "Summarize the main findings of research articles on effective business development models within suicide prevention voluntary organisations."),

        ("Ge en översikt av forskningsresultaten som beskriver bästa praxis för att integrera innovativa metoder och teknologier i självmordspreventiva insatser.",
         "Provide an overview of the research findings that describe best practices for integrating innovative methods and technologies into suicide prevention efforts."),

        ("Forskning om att bekämpa depression hos patienter med sjukdomshistoria",
         "Research on combating depression in patients with medical history")
        ]

results = []
for model in model_db_pairs:
    model_name = model[0]
    db_name = model[1]
    collection_name = 'mind'
    connection = f'postgresql+psycopg://postgres:password@localhost:5432/{db_name}'
    embedding = OllamaEmbeddings(model=model_name)
    vectorstore = PGVector(
            embeddings=embedding,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
            )
    for query in queries:
        q_sv = query[0]
        q_en = query[1]
        docs = vectorstore.similarity_search(q_en, k=5)
        results.append({
            'embedding_model': model_name,
            'sv_query': q_sv,
            'en_query': q_en,
            'page_content0': docs[0].page_content,
            'page_content1': docs[1].page_content,
            'page_content2': docs[2].page_content,
            'page_content3': docs[3].page_content,
            'page_content4': docs[4].page_content,
            })

df = pd.DataFrame(results)
df.to_csv('./test_queries_result.csv')
