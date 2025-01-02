# INTRODUCTION
Software integrity tools and automation is one of the fast evolving fields in the SDLC. I have had the pleasure to work in this field for years in my career. And have worked on the plethora of tools that are available and have integrated them for ease of use. One of the crucial steps in this effort is to: make these tools as part of the CI/CD pipeline. A natural progression towards breaking the entry barrier for the development teams across companies.
Even though it can be a challenge to achieve, software integrity can have profound benefits for your code. Because it ensures that:
- Coding defects are identified and addressed sooner.
- Code remains agile, modular, and maintainable.
- Cyber-threats are mitigated.
- Code development is well documented and organized.
- Code regulations and industry standards are met.
- Analytics are clear, detailed, and accessible.


I was thrilled to hear about the proposed capstone project in the Gen AI course, about using the AI models to conduct the PR review. And take the prompt engineered phase further to further automate the entire process, as the goal of the project.


# PROMPT ENGINEER PHASE
Experiment with the prompt that gives a near accurate result for reviewing the pull request. Turns out providing the git diff results in more accurate results overall.



`Could you describe briefly errors, issues or unhandled exceptions for the next code with given git diffs? Please, also, do not add intro words, just print errors in the format: "line_number : cause effect" If there are no errors, issues or unhandled exceptions just say "Looks Good to Me' (LGTM)". 
DIFFS: 
{diffs}`


# AUTOMATION PHASE
The following steps are automated via python script and Github action workflow setup stript.
Use GitHub Actions workflow
Runners are the machines that execute jobs in a GitHub Actions workflow. For example, a runner can clone your repository locally, install testing software, and then run commands that evaluate your code.
A self-hosted runner is a system that you deploy and manage to execute jobs from GitHub Actions on GitHub.
Set the workflow to run on pull request related events:

Setup python, install dependencies, checkout the code base, via the github workflow yaml file
https://github.com/random-droid/recursion/blob/main/.github/workflows/ai_pr_reviewer_workflow.yml
Execute the python script to run the pr reviewer in the VM
Make sure the required external parameters are setup:
CHATGPT_KEY - key for the intended AI model
CHATGPT_MODEL - the model to use among the available.  For example: gpt-4o-mini, gpt-3.5-turbo

Obtain list of files that have changed
`git diff –name-only base_ref head_ref`
Obtain the diff between the base ref and head ref in the PR for each file:
git diff base_ref head_ref — file_path
Create a prompt for OpenAI, instructing it to review the code based on the provided context.
	Insert the git diff from step 5 to the experimented final prompt from above
Send a request to OpenAI with the above prompt:

`curl https://api.openai.com/v1/chat/completions \ -H "Content-Type: application/json" \ -H "Authorization: Bearer $OPENAI_API_KEY" \ -d '{ "model": "gpt-4o-mini", "messages": [{"role": "user", "content": "the above prompt"}], "temperature": 0.7 }'`

Split the ai response and post them for the prescribed line and the file:

`curl -L \ -X POST \ -H "Accept: application/vnd.github+json" \ -H "Authorization: Bearer <YOUR-TOKEN>" \ -H "X-GitHub-Api-Version: 2022-11-28" \ https://api.github.com/repos/OWNER/REPO/pulls/PULL_NUMBER/comments \ -d '{"body":"text comment", "commit_id": commit_id, "path" : file_path, "position" : line}'`


# PROCEDURE
- Create a code base for running the PR review on, and push the base version to the master branch.
- Checkin the GitHub workflow with the workflow script in the .github folder within the code base.
- Checking the Python script and its related modules in the .genai folder within the code base
- Create another branch with the modification to the code base and push the modifications that you want to be reviewed.

# RESULTS
The model version makes a difference in the quality of the output. Based on the experimentation, gpt-4o-mini is more reliable over the older models.
The latency was not an issue in the small code base that was experimented on.

# CONCLUSION
While the PR review comments seem satisfactory and made logical sense. There is room to further enhance the idea. By setting the temperature lower. 
Have a RAG model to accommodate a Vector DB to hold  the context of the entire code base as the context for the review of individual files. Having the entire code base as a context at the prompt will certainly exceed the parameter limitation. But experimenting with the RAG to add the larger repeatable context may give more comprehensive semantics based review comments.

