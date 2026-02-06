"""
Multi-Agent System Extractor & Executor.

A Gradio web application that uses OpenAI's Vision API to convert hand-drawn
diagrams of multi-agent systems into structured JSON, then executes them using
CrewAI.

Features:
    - Image upload or webcam capture for diagrams
    - AI-powered diagram analysis
    - Configurable shape conventions for diagram interpretation
    - CrewAI-powered agent execution
    - Observability with metrics and logging
    - JSON export/import for system definitions

Usage:
    python app.py

Environment Variables:
    OPENAI_API_KEY: Required for extraction and execution
    DEFAULT_VISION_MODEL: Vision model for extraction (default: gpt-4o)
    DEFAULT_EXECUTION_MODEL: Model for agent execution (default: gpt-4o-mini)

Example:
    >>> from app import extract_system_from_image, execute_system
    >>> system = extract_system_from_image("diagram.png", "ovals=agents", api_key)
    >>> result = execute_system(system, "analyze this", api_key)
"""

# Standard imports
import base64
import json
import os
import logging

from datetime import datetime
from typing import Dict, Any

# Library imports
import gradio as gr

from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Default models (can be overridden via environment variables)
DEFAULT_VISION_MODEL = os.getenv("DEFAULT_VISION_MODEL", "gpt-4o")
DEFAULT_EXECUTION_MODEL = os.getenv("DEFAULT_EXECUTION_MODEL", "gpt-4o-mini")

EXTRACTION_PROMPT_TEMPLATE = """
You are an expert at analyzing hand-drawn diagrams of multi-agent systems
and converting them to structured JSON.

The user has drawn a diagram with the following conventions:
{shape_description}

Please analyze this diagram carefully and extract:
1. All agents in the system
2. Their roles
3. Their tools
4. The tasks each tool performs
5. The expected outputs

Return a JSON object with the following structure:

{{
    "agents": [
        {{
            "name": "Agent Name",
            "role": "Agent Role Description",
            "tools": [
            {{
                "name": "Tool Name",
                "tasks": ["task1", "task2"],
                "description": "What this tool does"
            }}
            ],
            "output": "Expected output description",
            "dependencies": ["other_agent_names"]
        }}
    ],
    "workflow": {{
        "description": "Overall workflow description",
        "steps": ["step1", "step2", "..."]
    }}
}}

Important:
- Extract exact text from the diagram as much as possible
- Infer connections between agents based on arrows/lines
- If text is unclear, make reasonable interpretations
- Ensure all agent names are consistent
- Include dependencies based on visual connections

Please analyze the image and return ONLY the JSON object, no other text.
"""

def encode_image(image_path: str) -> str:
    """Encode image to base64 string.

    Args:
        image_path: Path to the image file to encode.

    Returns:
        Base64 encoded string of the image content.

    Raises:
        FileNotFoundError: If the image file does not exist.
        IOError: If the file cannot be read.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_system_from_image(
    image_file_path: str,
    shape_desc: str,
    api_key: str,
    model: str = "gpt-4o"
) -> dict:
    """
    Extract multi-agent system structure from image using OpenAI Vision API

    Args:
        image_file_path: Path to the hand-drawn diagram image
        shape_desc: User's description of what shapes represent
        api_key: OpenAI API key
        model: Vision model to use (gpt-4o, gpt-4o-mini, gpt-4-turbo)

    Returns:
        Dictionary containing extracted system structure
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Encode image
    base64_image = encode_image(image_file_path)

    # Create prompt
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(shape_description=shape_desc)

    # Call OpenAI Vision API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0.2
        )

        # Extract JSON from response
        content = response.choices[0].message.content or ""

        # Try to parse as JSON
        # Sometimes the model wraps it in ```json ... ```
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()

        result = json.loads(json_str)

        # Add metadata
        result["_metadata"] = {
            "model": model,
            "extraction_timestamp": None,  # Can add timestamp if needed
            "shape_conventions": shape_desc
        }

        return result

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from model response: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Error during extraction: {str(e)}") from e

def validate_system_json(system_json: dict) -> tuple[bool, str]:  # pylint: disable=too-many-return-statements
    """
    Validate the extracted system JSON structure.

    Performs comprehensive validation of the multi-agent system structure,
    including agents, tools, and workflow definitions.

    Args:
        system_json: Dictionary containing the extracted system structure.

    Returns:
        A tuple of (is_valid, message) where is_valid is True if the
        structure is valid, and message contains either "Valid" or an
        error description.
    """
    if not isinstance(system_json, dict):
        return False, "System JSON must be a dictionary"

    if "agents" not in system_json:
        return False, "Missing 'agents' key in system JSON"

    if not isinstance(system_json["agents"], list):
        return False, "'agents' must be a list"

    if len(system_json["agents"]) == 0:
        return False, "'agents' list cannot be empty"

    agent_names = set()
    for i, agent in enumerate(system_json["agents"]):
        if "name" not in agent:
            return False, f"Agent {i} missing 'name' field"
        if "role" not in agent:
            return False, f"Agent {i} ({agent.get('name', 'unknown')}) missing 'role' field"

        # Track agent names for dependency validation
        agent_names.add(agent.get("name"))

        # Validate tools structure if present
        if "tools" in agent and not isinstance(agent["tools"], list):
            return False, f"Agent '{agent['name']}' tools must be a list"

    # Validate dependencies reference existing agents
    for agent in system_json["agents"]:
        dependencies = agent.get("dependencies", [])
        for dep in dependencies:
            if dep not in agent_names:
                return False, f"Agent '{agent['name']}' has unknown dependency: '{dep}'"

    return True, "Valid"


def execute_system(
    system_json_data: Dict[str, Any],
    query_text: str,
    api_key: str
) -> str:
    """
    Execute a multi-agent system using CrewAI

    Args:
        system_json_data: Extracted system structure
        query_text: User query to execute
        api_key: OpenAI API key

    Returns:
        Execution result as string
    """
    logger.info("Starting execution with CrewAI")
    logger.info("Query: %s", query_text)

    start_time = datetime.now()

    try:
        result = execute_with_crewai(system_json_data, query_text, api_key)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Save metrics
        metrics = {
            "framework": "CrewAI",
            "duration_seconds": duration,
            "timestamp": start_time.isoformat(),
            "query": query_text,
            "num_agents": len(system_json_data.get("agents", [])),
            "success": True
        }

        with open("metrics.json", "w", encoding='utf-8') as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

        logger.info("Execution completed in %.2f seconds", duration)

        return result

    except Exception as e:
        logger.error("Execution failed: %s", str(e), exc_info=True)

        # Save error metrics
        metrics = {
            "framework": "CrewAI",
            "timestamp": start_time.isoformat(),
            "query": query_text,
            "success": False,
            "error": str(e)
        }

        with open("metrics.json", "w", encoding='utf-8') as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

        raise

def execute_with_crewai(
    system_json_data: Dict[str, Any],
    query_text: str,
    api_key: str
) -> str:
    # pylint: disable=too-many-locals
    """Execute using CrewAI framework"""
    try:
        # pylint: disable=import-outside-toplevel
        import langchain

        from crewai import Agent, Task, Crew, Process
        from langchain_openai import ChatOpenAI

        # Patch stupid LangChain bug
        # https://github.com/langchain-ai/langchain/issues/4164
        langchain.verbose = False  # pyright: ignore[reportAttributeAccessIssue]

        # Set environment variable for API key (required by LangChain)
        os.environ["OPENAI_API_KEY"] = api_key

        logger.info("Initializing CrewAI agents...")

        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )

        # Create agents from JSON
        agents = []
        tasks = []

        for agent_config in system_json_data.get("agents", []):
            # Create agent
            agent = Agent(
                role=agent_config.get("role", "Assistant"),
                goal=f"Execute tasks related to: {agent_config.get('name', 'Unknown')}",
                backstory=f"An AI agent specialized in {agent_config.get('role', 'assistance')}",
                verbose=True,
                allow_delegation=True,
                llm=llm
            )
            agents.append(agent)

            # Create tasks for this agent
            tools_info = agent_config.get("tools", [])
            for tool in tools_info:
                task_description = f"{tool.get('description', tool.get('name', 'Task'))}"
                if tool.get('tasks'):
                    task_description += f"\nSubtasks: {', '.join(tool.get('tasks', []))}"

                task = Task(
                    description=task_description,
                    agent=agent,
                    expected_output=agent_config.get("output", "Completed task result")
                )
                tasks.append(task)

        # If no specific tasks, create a general task
        if not tasks and agents:
            task = Task(
                description=query_text,
                agent=agents[0],
                expected_output="Response to the user query"
            )
            tasks.append(task)

        # Create and run crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        logger.info("Running CrewAI crew...")
        crew_result = crew.kickoff()
        return str(crew_result)

    except ImportError:
        return ("❌ CrewAI not installed. "
                "Install with: pip install crewai langchain-openai")
    except Exception as e:
        logger.error("CrewAI execution error: %s", str(e))
        raise


def generate_script(system_json_data: Dict[str, Any], query_text: str) -> str:  # pylint: disable=too-many-locals
    """
    Generate a standalone Python script to run the multi-agent system with CrewAI.

    Args:
        system_json_data: Extracted system structure
        query_text: User query to execute

    Returns:
        Python script as a string
    """
    agents_code = []
    tasks_code = []

    for i, agent_config in enumerate(system_json_data.get("agents", [])):
        agent_name = agent_config.get("name", f"Agent{i}").replace(" ", "_").lower()
        role = agent_config.get("role", "Assistant")
        output = agent_config.get("output", "Completed task result")

        # Generate agent code
        agents_code.append(f'''
{agent_name} = Agent(
    role="{role}",
    goal="Execute tasks related to: {agent_config.get('name', 'Unknown')}",
    backstory="An AI agent specialized in {role.lower()}",
    verbose=True,
    allow_delegation=True,
    llm=llm
)''')

        # Generate tasks from tools
        tools_info = agent_config.get("tools", [])
        for j, tool in enumerate(tools_info):
            task_name = f"{agent_name}_task_{j}"
            task_description = tool.get("description", tool.get("name", "Task"))
            subtasks = tool.get("tasks", [])
            if subtasks:
                task_description += f"\\nSubtasks: {', '.join(subtasks)}"

            tasks_code.append(f'''
{task_name} = Task(
    description="""{task_description}""",
    agent={agent_name},
    expected_output="{output}"
)''')

    # Build the agent list and task list strings
    agent_names = [a.get("name", f"Agent{i}").replace(" ", "_").lower()
                   for i, a in enumerate(system_json_data.get("agents", []))]
    task_names = []
    for i, agent_config in enumerate(system_json_data.get("agents", [])):
        agent_name = agent_config.get("name", f"Agent{i}").replace(" ", "_").lower()
        for j, _ in enumerate(agent_config.get("tools", [])):
            task_names.append(f"{agent_name}_task_{j}")

    # If no tasks, create a default task
    default_task = ""
    if not task_names and agent_names:
        default_task = f'''
# Default task (no specific tasks defined in the system)
default_task = Task(
    description="""{query_text}""",
    agent={agent_names[0]},
    expected_output="Response to the user query"
)'''
        task_names = ["default_task"]

    script = f'''#!/usr/bin/env python3
"""
CrewAI Multi-Agent System

Generated from extracted system JSON.
Run with: python crewai_system.py
"""

import os

import langchain

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

langchain.verbose = False

# Ensure API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# =============================================================================
# AGENTS
# =============================================================================
{"\n".join(agents_code)}

# =============================================================================
# TASKS
# =============================================================================
{"\n".join(tasks_code)}{default_task}

# =============================================================================
# CREW
# =============================================================================

crew = Crew(
    agents=[
        {',\n\t\t'.join(agent_names)}
    ],
    tasks=[
        {',\n\t\t'.join(task_names)}
    ],
    process=Process.sequential,
    verbose=True
)

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)

'''

    return script


# Default example prompt
DEFAULT_PROMPT = """In my image:
- Ovals represent agents
- Squares inside the ovals represent the roles of the agents
- Rounded squares represent the tools of the agents
- Circles represent the tasks of the tools
- Diamond shapes represent the output of the agents"""

# Example prompts for reference
EXAMPLE_PROMPTS = {
    "Standard Convention": DEFAULT_PROMPT,
    "Alternative 1": """In my diagram:
- Rectangles are agents
- Circles inside rectangles are their roles
- Hexagons are tools
- Ovals are tasks
- Triangles are outputs""",
    "Alternative 2": """My conventions:
- Large boxes = agents
- Small boxes = roles
- Rounded boxes = tools
- Dots = tasks
- Stars = outputs"""
}

def extract_tab():
    """Create the Extract System tab"""
    with gr.Column():
        gr.Markdown("## 📸 Step 1: Upload Diagram")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload hand-drawn diagram",
                    type="filepath",
                    sources=["upload", "webcam"],
                    webcam_options=gr.WebcamOptions(
                        mirror=False,
                    )
                )
                gr.Markdown("**Supported:** `PNG`, `JPG`, `JPEG` (Max 200MB)")

            with gr.Column(scale=1):
                gr.Markdown("## ✏️ Step 2: Describe Shapes")

                with gr.Accordion("📋 View Example Prompts", open=False):
                    for name, prompt in EXAMPLE_PROMPTS.items():
                        gr.Markdown(f"**{name}:**\n```\n{prompt}\n```")

                shape_desc = gr.Textbox(
                    label="Describe your shape conventions",
                    value=DEFAULT_PROMPT,
                    lines=8,
                    placeholder="Tell us what each shape represents..."
                )

                gr.Markdown("✅ Prompt provided", elem_classes=["success-message"])

        gr.Markdown("## 🤖 Step 3: Extract System")

        with gr.Row():
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-...",
                value=os.getenv("OPENAI_API_KEY", "")
            )
            model_choice = gr.Dropdown(
                choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                value="gpt-4o",
                label="Model"
            )

        extract_button = gr.Button("🔍 Extract", variant="primary", size="lg")

        with gr.Row():
            extraction_status = gr.Textbox(label="Status", lines=2, interactive=False)

        with gr.Row():
            json_output = gr.JSON(label="Extracted System JSON")
            json_download = gr.File(label="Download JSON")

    # Handle extraction
    def handle_extraction(image, description, api_key, model):
        if not image:
            return "❌ Please upload an image", None, None
        if not description.strip():
            return "❌ Please provide shape descriptions", None, None
        if not api_key.strip():
            return "❌ Please provide OpenAI API Key", None, None

        try:
            result = extract_system_from_image(image, description, api_key, model)

            # Save JSON to file
            json_path = "extracted_system.json"
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2)

            return (
                "✅ System extracted successfully!",
                result,
                json_path
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"❌ Error: {str(e)}", None, None

    extract_button.click(  # pylint: disable=no-member
        fn=handle_extraction,
        inputs=[image_input, shape_desc, api_key_input, model_choice],
        outputs=[extraction_status, json_output, json_download]
    )

    return {
        "json_output": json_output,
        "api_key": api_key_input
    }


def execute_tab(shared_state):
    """Create the Execute Agents tab"""
    with gr.Column():
        gr.Markdown("## 📝 Step 4: Run Query")

        with gr.Row():
            with gr.Column():
                json_input = gr.JSON(label="System JSON (from Extract tab)")
                upload_json = gr.File(label="Or upload JSON file", file_types=[".json"])

            with gr.Column():
                query_input = gr.Textbox(
                    label="Enter your query",
                    placeholder="What would you like the agents to do?",
                    lines=4
                )

                execution_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    value=os.getenv("OPENAI_API_KEY", "")
                )

        execute_button = gr.Button("🚀 Run", variant="primary", size="lg")
        generate_script_btn = gr.Button("📄 Generate Script", variant="secondary", size="lg")

        with gr.Row():
            execution_status = gr.Textbox(label="Execution Status", lines=3, interactive=False)

        with gr.Row():
            execution_output = gr.Textbox(label="Results", lines=15, interactive=False)

        with gr.Row():
            script_output = gr.Code(
                label="Generated CrewAI Script",
                language="python",
                lines=20,
                interactive=False,
                visible=False
            )
            script_download = gr.File(label="Download Script", visible=False)

    # Handle JSON file upload
    def load_json_file(file):
        if file:
            with open(file.name, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    upload_json.change(  # pylint: disable=no-member
        fn=load_json_file,
        inputs=[upload_json],
        outputs=[json_input]
    )

    # Handle execution
    def handle_execution(json_data, query, api_key):
        if not json_data:
            return "❌ Please provide system JSON", ""
        if not query.strip():
            return "❌ Please enter a query", ""
        if not api_key.strip():
            return "❌ Please provide OpenAI API Key", ""

        try:
            result = execute_system(json_data, query, api_key)
            return "✅ Execution completed!", result
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"❌ Error: {str(e)}", ""

    execute_button.click(  # pylint: disable=no-member
        fn=handle_execution,
        inputs=[json_input, query_input, execution_api_key],
        outputs=[execution_status, execution_output]
    )

    # Handle script generation
    def handle_generate_script(json_data, query):
        if not json_data:
            return (
                gr.update(visible=True, value="# Error: Please provide system JSON first"),
                gr.update(visible=False)
            )

        try:
            script = generate_script(json_data, query or "Execute the multi-agent workflow")

            # Save script to file
            script_path = "crewai_system.py"
            with open(script_path, "w", encoding='utf-8') as f:
                f.write(script)

            return (
                gr.update(visible=True, value=script),
                gr.update(visible=True, value=script_path)
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            return (
                gr.update(visible=True, value=f"# Error generating script: {str(e)}"),
                gr.update(visible=False)
            )

    generate_script_btn.click(  # pylint: disable=no-member
        fn=handle_generate_script,
        inputs=[json_input, query_input],
        outputs=[script_output, script_download]
    )

    # Sync JSON from extract tab
    if "json_output" in shared_state:
        shared_state["json_output"].change(  # pylint: disable=no-member
            fn=lambda x: x,
            inputs=[shared_state["json_output"]],
            outputs=[json_input]
        )
        shared_state["api_key"].change(  # pylint: disable=no-member
            fn=lambda x: x,
            inputs=[shared_state["api_key"]],
            outputs=[execution_api_key]
        )

def analyze_results_tab():
    """Create the Analyze Results tab"""
    with gr.Column():
        gr.Markdown("## 📊 Step 5: Analyze Results")

        with gr.Row():
            refresh_button = gr.Button("🔄 Refresh", variant="secondary")
            show_last_n = gr.Slider(
                minimum=10,
                maximum=200,
                value=50,
                step=10,
                label="Show last N log lines"
            )
            newest_first = gr.Checkbox(label="Newest first", value=True)

        with gr.Row():
            with gr.Column():
                metrics_display = gr.JSON(label="System Metrics")

            with gr.Column():
                logs_display = gr.Code(
                    label="Execution Logs",
                    language=None,
                    lines=20,
                    interactive=False
                )

        download_logs = gr.Button("💾 Download")
        logs_file = gr.File(label="Log File")

        def format_logs(log_content: str, last_n: int, reverse: bool) -> str:
            """Format log content for display."""
            if not log_content:
                return "No logs available yet!"

            lines = log_content.strip().split('\n')

            # Take last N lines
            if len(lines) > last_n:
                lines = lines[-last_n:]

            # Reverse if newest first
            if reverse:
                lines = lines[::-1]

            return '\n'.join(lines)

        def refresh_metrics(last_n, reverse):
            # Load metrics if available
            try:
                if os.path.exists("metrics.json"):
                    with open("metrics.json", "r", encoding='utf-8') as f:
                        metrics = json.load(f)
                else:
                    metrics = {"status": "No execution data yet"}

                if os.path.exists("execution.log"):
                    with open("execution.log", "r", encoding='utf-8') as f:
                        raw_logs = f.read()
                    logs = format_logs(raw_logs, int(last_n), reverse)
                else:
                    logs = "No logs available yet"

                return metrics, logs
            except Exception:  # pylint: disable=broad-exception-caught
                return {"error": "Failed to load metrics"}, ""

        def save_logs():
            if os.path.exists("execution.log"):
                return "execution.log"
            return None

        refresh_button.click(  # pylint: disable=no-member
            fn=refresh_metrics,
            inputs=[show_last_n, newest_first],
            outputs=[metrics_display, logs_display]
        )

        download_logs.click(  # pylint: disable=no-member
            fn=save_logs,
            outputs=[logs_file]
        )

def create_app():
    """Create and configure the Gradio application"""
    with gr.Blocks(
        title="Diagram-to-Agents",
    ) as app:
        gr.Markdown("""
        # 🤖 Diagram-to-Agents
        ### Convert hand-drawn diagrams of multi-agent systems to JSON → Execute with CrewAI
        """)

        shared_state = {}

        with gr.Tabs():
            with gr.Tab("🎨 Extract System"):
                shared_state = extract_tab()

            with gr.Tab("🚀 Execute Agents"):
                execute_tab(shared_state)

            with gr.Tab("📊 Analyze Results"):
                analyze_results_tab()

    return app


def main():
    """Main entrypoint"""
    load_dotenv()
    app = create_app()
    app.launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()
