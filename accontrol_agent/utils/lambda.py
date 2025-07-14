import json
import boto3

def retrieve_knowledge(input_prompt: str):
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='ap-northeast-1')
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId='JOSLJLSFLZ',
        retrievalQuery={'text': input_prompt},
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': 10,
            }
        }
    )

    generated_text = response.get('output', {}).get('text', '')
    results = response['retrievalResults'] if response.get('retrievalResults') else []
    citations = []
    for idx, item in enumerate(results, 1):  # 1から始める
        content = item.get("content", {}).get("text", "")
        document_uri = item.get("location", {}).get("s3Location", {}).get("uri", "")
        pdf_name = document_uri.split("/")[-1].replace(".txt", ".pdf")
        title = item.get("title", pdf_name)
        citations.append({
            'index': idx,
            'content': content,
            'pdf_name': pdf_name,
            'title': title,
        })

    return {
        "citations": citations,
        "generated_text": generated_text
    }

def generate_response(prompt: str):
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    try:
        input_data = {
            "thinking": {"type": "enabled", "budget_tokens": 1600},
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "anthropic_version": "bedrock-2023-05-31"
        }
        response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            body=json.dumps(input_data),
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body
    except Exception as e:
        return {"error": str(e)}

def create_rag_prompt(input_prompt: str, knowledge_info: dict):
    citations = knowledge_info.get("citations", [])
    print("Citations", citations)

    citation_text = ""
    for c in citations:
        citation_text += f"[{c['index']}] {c['pdf_name']}\n"

    prompt = f"""
    あなたはGE社の優秀なアシスタントです。
    以下のデータに基づいて、ユーザーの質問に正確かつ端的に答えてください。
    なお、利用可能な情報はこのデータのみに限定してください。
    指定されたデータに基づいて回答できない場合は、無理に答えず、「質問を変えて質問してください。」と出力してください。
    回答の中で引用情報を参照した箇所の直後には、[1]や[2]のような番号を付けてください。その番号は参考したマニュアルの順番になります。
    参考したその番号とマニュアル名を一番下にリストしてください。
    

    質問：{input_prompt}

　　利用可能なマニュアル情報：{str(citations)}
    
    """
    return {"prompt": prompt, "citation_text": citation_text}

def lambda_handler(event, context):
    input_prompt = event.get("prompt", "")

    knowledge_info = retrieve_knowledge(input_prompt)
    rag_info = create_rag_prompt(input_prompt, knowledge_info)
    rag_prompt = rag_info.get("prompt")

    response = generate_response(rag_prompt)

    answer_text = ""
    if isinstance(response, dict):
        answer_text = response.get('content', [{}])[0].get('text', '')

    citation_text = rag_info.get("citation_text")

    citations = rag_info.get("citations")
    # print("----------citation_text----------", citation_text)
    # print("----------answer_text----------", answer_text)

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET',
            'Content-Type': 'application/json'
        },
        'body': json.dumps({
            "response": answer_text,
            "citation_text": citation_text,
            "citations": citations
            })
    }
