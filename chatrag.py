from dataclasses import dataclass
import ollama

SYSTEM_PROMPT_EN="""
      Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context:
     {context}
     """


SYSTEM_PROMPT_CZ="""
     K odpovědi na otázku uživatele použijte následující části kontextu. Pokud neznáte odpověď, řekněte jen, že nevíte, nesnažte se odpověď vymýšlet. Kontext:
     {context}
     """

@dataclass
class ChatConfig:
     model="llama3.2:3b"
    #    original_system_prompt = 
    #     """
    # Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context:
    # {context}      
    # """
    
    # chttps://pdx.www.deepl.com/en/translator
    #     K zodpovězení otázky uživatele použijte následující souvislosti. Pokud odpověď neznáte, prostě řekněte, že nevíte, nesnažte se odpověď vymyslet. Kontext: 
    # https://translate.google.com/ 
    # K odpovědi na otázku uživatele použijte následující části kontextu. Pokud neznáte odpověď, řekněte jen, že nevíte, nesnažte se odpověď vymýšlet. Kontext:
     original_system_prompt = SYSTEM_PROMPT_EN
# You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

# context will be passed as "Context:"
# user question will be passed as "Question:"

# To answer the question:
# 1. Thoroughly analyze the context, identifying key information relevant to the question.
# 2. Organize your thoughts and plan your response to ensure a logical flow of information.
# 3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
# 4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
# 5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

# Format your response as follows:
# 1. Use clear, concise language.
# 2. Organize your answer into paragraphs for readability.
# 3. Use bullet points or numbered lists where appropriate to break down complex information.
# 4. If relevant, include any headings or subheadings to structure your response.
# 5. Ensure proper grammar, punctuation, and spelling throughout your answer.

# Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
# """


class ChatRag:
    config=ChatConfig()
    # TODO: put me in config    

    def format_prompts(self,context: list[str], prompt: str,system_prompt:str):
        return [
                {
                    "role": "system",
                    "content": system_prompt.replace("{context}",context),
                },
                {
                    "role": "user",
                    "content": prompt.replace("{context}",context),
                },
            ]
    def call_llm(self,prompts):
        """Calls the language model with context and prompt to generate a response.

        Uses Ollama to stream responses from a language model by providing context and a
        question prompt. The model uses a system prompt to format and ground its responses appropriately.

        Args:
            context: String containing the relevant context for answering the question
            prompt: String containing the user's question

        Yields:
            String chunks of the generated response as they become available from the model

        Raises:
            OllamaError: If there are issues communicating with the Ollama API
        """
        response = ollama.chat(
            model=self.config.model,
            stream=True,
            messages=prompts
        )
        for chunk in response:
            if chunk["done"] is False:
                yield chunk["message"]["content"]
            else:
                break
                                