import pandas as pd
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains import SequentialChain
from tqdm import tqdm


default_instruction = """Your task is to solve a given mystery.
The mystery is a detective puzzle presented as a short story.
You will be given a list of answer options apart from the mystery content. 
Please give your final answer as
(x) Your Answer
where x is the number of the answer option.
Only one answer from the list is correct, and your task is to identify which one.\n\n\n"""

default_mistery_body = """Answer options: {suspects}.

Mystery content:
{mystery_name}

{mystery_content}"""

default_stepbystep = """\n\nFull answer: 
Let's think step by step."""

default_outcome = """\n\nSolution: 
{outcome}"""

default_final_q = """\n\nFinal answer:"""


def same_answers(pred_a: str, true_a: str):
    if pred_a != true_a:
        # discard dot at the end of answers
        pred_a, true_a = strip_answers(pred_a, true_a)

    return int(pred_a == true_a)


def strip_answers(pred_a, true_a):
    pred_a = pred_a[:-1] if pred_a[-1] == "." else pred_a
    true_a = true_a[:-1] if true_a[-1] == "." else true_a

    # discard (x) at the beginning of answers
    pred_a = pred_a[3:]
    true_a = true_a[3:]
    return pred_a, true_a


def compute_solve_rate(pred_answers, true_answers):
    solve_rate = 0
    for pred_a, true_a in zip(pred_answers, true_answers):
        if same_answers(pred_a, true_a):
            solve_rate += 1
    return solve_rate / len(pred_answers)


def random_baseline(data_path="detective-puzzles.csv"):
    # iterate over all cases and compute solve rate of random baseline
    # random baseline: randomly choose one of the answer options
    # make 10 random restarts
    import random

    # set seed
    random.seed(69)

    df = pd.read_csv(data_path)

    accuracy_per_restart = []
    for restart in range(256):
        random_solve_rate_per_case = []
        for i in range(len(df)):
            answer_options = df["answer_options"][i].split("; ")
            random_answer = random.choice(answer_options)
            random_solve_rate_per_case.append(
                int(same_answers(random_answer, df["answer"][i]))
            )
        # get accuracy
        accuracy_per_restart.append(
            sum(random_solve_rate_per_case) / len(random_solve_rate_per_case)
        )
    # avg accuracy
    avg = sum(accuracy_per_restart) / len(accuracy_per_restart)
    print(f"random baseline accuracy: {avg}")
    return avg


def save_answers(model_name, output_folder, output_file, df_pred):
    fn = model_name + "_" + output_file
    df_pred.to_csv(f"{output_folder}/{fn}", index=False)
    print(f"saved predictions to {output_folder}/{fn}")


def eval_vanilla(
    model_name="text-davinci-003",
    data_path="detective-puzzles.csv",
    output_folder="eval_results",
    output_file="instruct_vanilla.csv",
    instruction=default_instruction,
    mystery_body=default_mistery_body,
    final_q=default_final_q,
):
    llm = OpenAI(
        model_name=model_name,
        temperature=0,
        max_tokens=64,
    )

    template = instruction + mystery_body + final_q
    print(template)

    answer_chain = LLMChain(
        llm=llm,
        verbose=False,
        output_key="answer",
        prompt=PromptTemplate(
            template=template,
            input_variables=["suspects", "mystery_name", "mystery_content"],
        ),
    )

    predictions_answers = []
    # predictions_chain_of_thought = []
    df = pd.read_csv(data_path)

    for i in tqdm(range(len(df))):
        pred = answer_chain(
            {
                "suspects": df["answer_options"][i],
                "mystery_name": df["case_name"][i],
                "mystery_content": df["mystery_text"][i],
            }
        )

        predictions_answers.append(pred["answer"].strip())

    # save predictions
    df_pred = pd.DataFrame({"answer": predictions_answers})
    save_answers(model_name, output_folder, output_file, df_pred)
    solve_rate = compute_solve_rate(df_pred["answer"], df["answer"])
    print(f"solve rate: {solve_rate}")
    return df_pred, solve_rate


def eval_step_by_step(
    model_name="text-davinci-003",
    data_path="detective-puzzles.csv",
    output_folder="eval_results",
    output_file="instruct_step-by-step.csv",
    instruction=default_instruction,
    mystery_body=default_mistery_body,
    stepbystep=default_stepbystep,
    final_q=default_final_q,
):

    template_1 = instruction + mystery_body + stepbystep
    template_2 = template_1 + "{chain_of_thought}" + final_q

    print(template_2)

    llm = OpenAI(
        model_name=model_name,
        temperature=0,
        max_tokens=512,
    )

    cot_chain = LLMChain(
        llm=llm,
        verbose=False,
        output_key="chain_of_thought",
        prompt=PromptTemplate(
            template=template_1,
            input_variables=["suspects", "mystery_name", "mystery_content"],
        ),
    )

    llm = OpenAI(
        model_name=model_name,
        temperature=0,
        max_tokens=64,
    )

    answer_chain = LLMChain(
        llm=llm,
        verbose=False,
        output_key="answer",
        prompt=PromptTemplate(
            template=template_2,
            input_variables=[
                "suspects",
                "mystery_name",
                "mystery_content",
                "chain_of_thought",
            ],
        ),
    )

    # This is the overall chain where we run these two chains in sequence.

    overall_chain = SequentialChain(
        verbose=False,
        chains=[cot_chain, answer_chain],
        input_variables=["suspects", "mystery_name", "mystery_content"],
        output_variables=["chain_of_thought", "answer"],
    )

    # eval

    predictions_answers = []
    predictions_chain_of_thought = []
    df = pd.read_csv(data_path)

    for i in tqdm(range(len(df))):
        pred = overall_chain(
            {
                "suspects": df["answer_options"][i],
                "mystery_name": df["case_name"][i],
                "mystery_content": df["mystery_text"][i],
            }
        )

        predictions_answers.append(pred["answer"].strip())
        predictions_chain_of_thought.append(pred["chain_of_thought"])

    # save predictions
    df_pred = pd.DataFrame({"answer": predictions_answers})
    df_pred["chain_of_thought"] = predictions_chain_of_thought

    save_answers(model_name, output_folder, output_file, df_pred)
    solve_rate = compute_solve_rate(df_pred["answer"], df["answer"])
    print(f"solve rate: {solve_rate}")
    return df_pred, solve_rate


def eval_outcome(
    model_name="text-davinci-003",
    data_path="detective-puzzles.csv",
    output_folder="eval_results",
    output_file="instruct_outcome.csv",
    instruction=default_instruction,
    mystery_body=default_mistery_body,
    outcome=default_outcome,
    final_q=default_final_q,
):
    llm = OpenAI(
        model_name=model_name,
        temperature=0,
        max_tokens=64,
    )

    template = instruction + mystery_body + outcome + final_q
    print(template)

    answer_chain = LLMChain(
        llm=llm,
        verbose=False,
        output_key="answer",
        prompt=PromptTemplate(
            template=template,
            input_variables=[
                "suspects",
                "mystery_name",
                "mystery_content",
                "outcome",
            ],
        ),
    )

    predictions_answers = []
    # predictions_chain_of_thought = []
    df = pd.read_csv(data_path)

    for i in tqdm(range(len(df))):
        pred = answer_chain(
            {
                "suspects": df["answer_options"][i],
                "mystery_name": df["case_name"][i],
                "mystery_content": df["mystery_text"][i],
                "outcome": df["outcome"][i],
            }
        )

        predictions_answers.append(pred["answer"].strip())

    # save predictions
    df_pred = pd.DataFrame({"answer": predictions_answers})
    save_answers(model_name, output_folder, output_file, df_pred)
    solve_rate = compute_solve_rate(df_pred["answer"], df["answer"])
    print(f"solve rate: {solve_rate}")
    return df_pred, solve_rate


def eval_outcome_step_by_step(
    model_name="text-davinci-003",
    data_path="detective-puzzles.csv",
    output_folder="eval_results",
    output_file="instruct_outcome_step-by-step.csv",
    instruction=default_instruction,
    mystery_body=default_mistery_body,
    stepbystep=default_stepbystep,
    outcome=default_outcome,
    final_q=default_final_q,
):
    # step by step on top of full answer
    template_1 = instruction + mystery_body + outcome + stepbystep
    template_2 = template_1 + "{chain_of_thought}" + final_q

    print(template_2)

    llm = OpenAI(
        model_name=model_name,
        temperature=0,
        max_tokens=512,
    )

    cot_chain = LLMChain(
        llm=llm,
        verbose=False,
        output_key="chain_of_thought",
        prompt=PromptTemplate(
            template=template_1,
            input_variables=["suspects", "mystery_name", "mystery_content", "outcome"],
        ),
    )

    llm = OpenAI(
        model_name=model_name,
        temperature=0,
        max_tokens=64,
    )

    answer_chain = LLMChain(
        llm=llm,
        verbose=False,
        output_key="answer",
        prompt=PromptTemplate(
            template=template_2,
            input_variables=[
                "suspects",
                "mystery_name",
                "mystery_content",
                "outcome",
                "chain_of_thought",
            ],
        ),
    )

    # This is the overall chain where we run these two chains in sequence.

    overall_chain = SequentialChain(
        verbose=False,
        chains=[cot_chain, answer_chain],
        input_variables=["suspects", "mystery_name", "mystery_content", "outcome"],
        output_variables=["chain_of_thought", "answer"],
    )

    # eval

    predictions_answers = []
    predictions_chain_of_thought = []
    df = pd.read_csv(data_path)

    for i in tqdm(range(len(df))):
        pred = overall_chain(
            {
                "suspects": df["answer_options"][i],
                "mystery_name": df["case_name"][i],
                "mystery_content": df["mystery_text"][i],
                "outcome": df["outcome"][i],
            }
        )

        predictions_answers.append(pred["answer"].strip())
        predictions_chain_of_thought.append(pred["chain_of_thought"])

    # save predictions
    df_pred = pd.DataFrame({"answer": predictions_answers})
    df_pred["chain_of_thought"] = predictions_chain_of_thought

    save_answers(model_name, output_folder, output_file, df_pred)
    solve_rate = compute_solve_rate(df_pred["answer"], df["answer"])
    print(f"solve rate: {solve_rate}")
    return df_pred, solve_rate
