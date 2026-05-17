class BaseAgentCore:
    def __init__(self, name, role, llm=None):
        self.name = name
        self.role = role
        self.llm = llm

    def __call__(self, user_input):
        if self.llm is None:
            return user_input, None
        return self.llm(user_input), None

    def should_terminate(self, user_input):
        text = str(user_input).strip().lower()
        return text in {"quit", "exit", "stop", "done"}

    def summarize(self, user_input, response):
        return response


class AgentBase:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.core = BaseAgentCore(name, role)

    def build_context(self, user_input):
        return f"""
User: {user_input}
Use the following context to answer the user's query.
"""

    def generate_response(self, user_input):
        agent_input = self.build_context(user_input)
        response, extra = self.core(agent_input)
        return response, extra

    def __call__(self, user_input, caller="User"):
        print(f"{caller}: {user_input}")
        if self.core.should_terminate(user_input):
            print(f"{self.name}: Task completed.")
            return None

        response, _ = self.generate_response(user_input)
        summary = self.core.summarize(user_input, response)
        print(f"{self.name}: {summary}")
        return summary

    def run(self):
        while True:
            query = input("User Input: ")
            response = self(query, caller="User")
            if response is None:
                break
