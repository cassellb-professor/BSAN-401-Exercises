# Databricks notebook source
# MAGIC %md
# MAGIC # Quickstart: Build, test, and deploy an agent using Mosaic AI Agent Framework
# MAGIC This quickstart notebook demonstrates how to build, test, and deploy a generative AI agent ([AWS](https://docs.databricks.com/aws/en/generative-ai/guide/introduction-generative-ai-apps#what-are-gen-ai-apps) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/guide/introduction-generative-ai-apps#what-are-gen-ai-apps) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/guide/introduction-generative-ai-apps)) using Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/build-genai-apps#-mosaic-ai-agent-framework) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/build-genai-apps#-mosaic-ai-agent-framework) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/build-genai-apps#-mosaic-ai-agent-framework)) on Databricks

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Define and test an agent
# MAGIC This section defines and tests a simple agent with the following attributes:
# MAGIC
# MAGIC - The agent uses an LLM served on Databricks Foundation Model API ([AWS](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/) | [GCP](https://docs.databricks.com/gcp/en/machine-learning/foundation-model-apis))
# MAGIC - The agent has access to a single tool, the built-in Python code interpreter tool on Databricks Unity Catalog. It can use this tool to run LLM-generated code in order to respond to user questions. ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/code-interpreter-tools#built-in-python-executor-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/code-interpreter-tools) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/code-interpreter-tools))
# MAGIC
# MAGIC We will use `databricks_openai` SDK ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent#requirements) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent#requirements) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent#requirements)) to query the LLM endpoint.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-openai databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# The snippet below tries to pick the first LLM API available in your Databricks workspace
# from a set of candidates. You can override and simplify it
# to just specify LLM_ENDPOINT_NAME.
LLM_ENDPOINT_NAME = None

from databricks.sdk import WorkspaceClient
def is_endpoint_available(endpoint_name):
  try:
    client = WorkspaceClient().serving_endpoints.get_open_ai_client()
    client.chat.completions.create(model=endpoint_name, messages=[{"role": "user", "content": "What is AI?"}])
    return True
  except Exception:
    return False
  
client = WorkspaceClient()
for candidate_endpoint_name in ["databricks-claude-3-7-sonnet", "databricks-meta-llama-3-3-70b-instruct"]:
    if is_endpoint_available(candidate_endpoint_name):
      LLM_ENDPOINT_NAME = candidate_endpoint_name
assert LLM_ENDPOINT_NAME is not None, "Please specify LLM_ENDPOINT_NAME"

# COMMAND ----------

import json
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient

# Automatically log traces from LLM calls for ease of debugging
mlflow.openai.autolog()

# Get an OpenAI client configured to talk to Databricks model serving endpoints
# We'll use this to query an LLM in our agent
openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()

# Load Databricks built-in tools (a stateless Python code interpreter tool)
client = DatabricksFunctionClient()
builtin_tools = UCFunctionToolkit(
    function_names=["system.ai.python_exec"], client=client
).tools
for tool in builtin_tools:
    del tool["function"]["strict"]


def call_tool(tool_name, parameters):
    if tool_name == "system__ai__python_exec":
        return DatabricksFunctionClient().execute_function(
            "system.ai.python_exec", parameters=parameters
        )
    raise ValueError(f"Unknown tool: {tool_name}")


def run_agent(prompt):
    """
    Send a user prompt to the LLM, and return a list of LLM response messages
    The LLM is allowed to call the code interpreter tool if needed, to respond to the user
    """
    result_msgs = []
    response = openai_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=[{"role": "user", "content": prompt}],
        tools=builtin_tools,
    )
    msg = response.choices[0].message
    result_msgs.append(msg.to_dict())

    # If the model executed a tool, call it
    if msg.tool_calls:
        call = msg.tool_calls[0]
        tool_result = call_tool(call.function.name, json.loads(call.function.arguments))
        result_msgs.append(
            {
                "role": "tool",
                "content": tool_result.value,
                "name": call.function.name,
                "tool_call_id": call.id,
            }
        )
    return result_msgs

# COMMAND ----------

answer = run_agent("What is the square root of 429?")
for message in answer:
    print(f'{message["role"]}: {message["content"]}')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Prepare agent code for logging
# MAGIC
# MAGIC Wrap your agent definition in MLflow’s [ChatAgent interface](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) to prepare your code for logging.
# MAGIC
# MAGIC By using MLflow’s standard agent authoring interface, you get built-in UIs for chatting with your agent and sharing it with others after deployment. ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent#-use-chatagent-to-author-agents) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/author-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/author-agent))

# COMMAND ----------

import uuid
import mlflow
from typing import Any, Optional

from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext

mlflow.openai.autolog()

class QuickstartAgent(ChatAgent):
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # 1. Extract the last user prompt from the input messages
        prompt = messages[-1].content

        # 2. Call run_agent to get back a list of response messages
        raw_msgs = run_agent(prompt)

        # 3. Map each response message into a ChatAgentMessage and return
        # the response
        out = []
        for m in raw_msgs:
            out.append(ChatAgentMessage(id=uuid.uuid4().hex, **m))

        return ChatAgentResponse(messages=out)

# COMMAND ----------

AGENT = QuickstartAgent()
for response_message in AGENT.predict(
    {"messages": [{"role": "user", "content": "What's the square root of 429?"}]}
).messages:
    print(f"role: {response_message.role}, content: {response_message.content}")

# COMMAND ----------

# MAGIC %md ## Log the agent
# MAGIC
# MAGIC Log the agent and register it to Unity Catalog as a model ([AWS](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/) | [GCP](https://docs.databricks.com/gcp/en/machine-learning/manage-model-lifecycle/)). This step packages the agent code and its dependencies into a single artifact to deploy it to a serving endpoint.
# MAGIC
# MAGIC The following code cells do the following:
# MAGIC
# MAGIC 1. Copy the agent code from above and combine it into a single cell.
# MAGIC 1. Add the `%%writefile` cell magic command at the top of the cell to save the agent code to a file called `quickstart_agent.py`.
# MAGIC 1. Add a [mlflow.models.set_model()](https://mlflow.org/docs/latest/model#models-from-code) call to the bottom of the cell. This tells MLflow which Python agent object to use for making predictions when your agent is deployed.
# MAGIC 1. Log the agent code in the `quickstart_agent.py` file using MLflow APIs ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/log-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/log-agent)).

# COMMAND ----------

# MAGIC %%writefile quickstart_agent.py
# MAGIC
# MAGIC import json
# MAGIC import uuid
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient
# MAGIC from typing import Any, Optional
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext
# MAGIC
# MAGIC # Get an OpenAI client configured to talk to Databricks model serving endpoints
# MAGIC # We'll use this to query an LLM in our agent
# MAGIC openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC # The snippet below tries to pick the first LLM API available in your Databricks workspace
# MAGIC # from a set of candidates. You can override and simplify it
# MAGIC # to just specify LLM_ENDPOINT_NAME.
# MAGIC LLM_ENDPOINT_NAME = None
# MAGIC
# MAGIC def is_endpoint_available(endpoint_name):
# MAGIC   try:
# MAGIC     client = WorkspaceClient().serving_endpoints.get_open_ai_client()
# MAGIC     client.chat.completions.create(model=endpoint_name, messages=[{"role": "user", "content": "What is AI?"}])
# MAGIC     return True
# MAGIC   except Exception:
# MAGIC     return False
# MAGIC   
# MAGIC for candidate_endpoint_name in ["databricks-claude-3-7-sonnet", "databricks-meta-llama-3-3-70b-instruct"]:
# MAGIC     if is_endpoint_available(candidate_endpoint_name):
# MAGIC       LLM_ENDPOINT_NAME = candidate_endpoint_name
# MAGIC assert LLM_ENDPOINT_NAME is not None, "Please specify LLM_ENDPOINT_NAME"
# MAGIC
# MAGIC # Enable automatic tracing of LLM calls
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC # Load Databricks built-in tools (a stateless Python code interpreter tool)
# MAGIC client = DatabricksFunctionClient()
# MAGIC builtin_tools = UCFunctionToolkit(function_names=["system.ai.python_exec"], client=client).tools
# MAGIC for tool in builtin_tools:
# MAGIC     del tool["function"]["strict"]
# MAGIC
# MAGIC def call_tool(tool_name, parameters):
# MAGIC     if tool_name == "system__ai__python_exec":
# MAGIC         return DatabricksFunctionClient().execute_function("system.ai.python_exec", parameters=parameters)
# MAGIC     raise ValueError(f"Unknown tool: {tool_name}")
# MAGIC
# MAGIC def run_agent(prompt):
# MAGIC     """
# MAGIC     Send a user prompt to the LLM, and return a list of LLM response messages
# MAGIC     The LLM is allowed to call the code interpreter tool if needed, to respond to the user
# MAGIC     """
# MAGIC     result_msgs = []
# MAGIC     response = openai_client.chat.completions.create(
# MAGIC         model=LLM_ENDPOINT_NAME,
# MAGIC         messages=[{"role": "user", "content": prompt}],
# MAGIC         tools=builtin_tools,
# MAGIC     )
# MAGIC     msg = response.choices[0].message
# MAGIC     result_msgs.append(msg.to_dict())
# MAGIC
# MAGIC     # If the model executed a tool, call it
# MAGIC     if msg.tool_calls:
# MAGIC         call = msg.tool_calls[0]
# MAGIC         tool_result = call_tool(call.function.name, json.loads(call.function.arguments))
# MAGIC         result_msgs.append({"role": "tool", "content": tool_result.value, "name": call.function.name, "tool_call_id": call.id})
# MAGIC     return result_msgs
# MAGIC
# MAGIC class QuickstartAgent(ChatAgent):
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         prompt = messages[-1].content
# MAGIC         raw_msgs = run_agent(prompt)
# MAGIC         out = []
# MAGIC         for m in raw_msgs:
# MAGIC             out.append(ChatAgentMessage(
# MAGIC                 id=uuid.uuid4().hex,
# MAGIC                 **m
# MAGIC             ))
# MAGIC
# MAGIC         return ChatAgentResponse(messages=out)
# MAGIC
# MAGIC AGENT = QuickstartAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from pkg_resources import get_distribution
from quickstart_agent import LLM_ENDPOINT_NAME

# Register the model to the workspace default catalog.
# Specify a catalog (e.g. "main") and schema name (e.g. "custom_schema") if needed,
# in order to register the agent to a different location
catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]
schema_name = "default"
registered_model_name = f"{catalog_name}.{schema_name}.quickstart_agent"

# Specify Databricks product resources that the agent needs access to (our builtin python
# code interpreter tool and LLM serving endpoint), so that Databricks can automatically
# configure authentication for the agent to access these resources when it's deployed.
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksFunction(function_name="system.ai.python_exec"),
]

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="quickstart_agent.py",
        extra_pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}"
        ],
        resources=resources,
        registered_model_name=registered_model_name,
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Deploy the agent
# MAGIC
# MAGIC Run the cell below to deploy the agent ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/deploy-agent) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-framework/deploy-agent)). Once the agent endpoint starts, you can chat with it via AI Playground ([AWS](https://docs.databricks.com/aws/en/large-language-models/ai-playground) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/large-language-models/ai-playground) | [GCP](https://docs.databricks.com/gcp/en/large-language-models/ai-playground)), or share it with stakeholders ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/review-app) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/review-app) | [GCP](https://docs.databricks.com/gcp/en/generative-ai/agent-evaluation/review-app)) for initial feedback, before sharing it more broadly.

# COMMAND ----------

from databricks import agents

deployment_info = agents.deploy(
    model_name=registered_model_name,
    model_version=int(logged_agent_info.version),
    workspace_url=None # Add workspace_url if required by your environment
)
