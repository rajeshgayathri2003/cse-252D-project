## Visual Navigation via Agentic systems in ProcTHOR

### Installation
Run the following commands to install necessary packages:

```
pip install procthor
pip install --upgrade ai2thor
pip install prior
```

### Testing the Installation
Use `python3 test_2.py` to test your installation. 

### Setting your API Key
Run `export TRITONAI_API_KEY="your-api-key"` or `export OPENAI_API_KEY="your-api-key"`
Check whether the key has been added to your environment using `echo TRITONAI_API_KEY` or `echo OPENAI_API_KEY`

### Run the pipeline
To run the full visual navigation pipeline with in-context examples, run:
```bash
python3 run_navigation_pipeline.py --tritonai --eval-tasks eval_scenes/eval_tasks.txt --steps 50 --in-context 
```
To run the full pipeline (zero-shot), run:

```bash
python3 run_navigation_pipeline.py --tritonai --eval-tasks eval_scenes/eval_tasks.txt --steps 50  
```

To start from a particular task use:
```bash
python3 run_navigation_pipeline.py --tritonai --eval-tasks eval_tasks_run.txt --start-from 720:T3 --steps 50
```

