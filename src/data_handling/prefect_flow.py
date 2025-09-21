import os
import prefect
from prefect import flow, task, serve
from prefect.deployments import Deployment

from prefect_lakefs import lakefs_commit # This is a placeholder for a real Prefect lakeFS integration

# Import the user-provided data processing scripts
# Note: These scripts need to be accessible in the environment where the flow runs.

import data_handling as dh
from src._utils import main_logger


# aws
from prefect_aws import AwsCredentials
from prefect_aws.glue_job import GlueJobBlock, GlueJobRun


def glue_job(stockcode: str = ''):
    import data_handling as dh

    if stockcode:
        X_train, X_test, y_train, y_test, preprocessor = dh.main_script_by_stockcode(stockcode)
        return X_train, X_test, y_train, y_test, preprocessor
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = dh.main_script()
        return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor



BLOCK_NAME='test'

@flow
def run_glue_job_data_processing(stockcode: str = ''):
    AwsCredentials(
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'), # type: ignore
        region_name=os.environ.get('AWS_REGION_NAME'),
    ).save(BLOCK_NAME)

    glue_job_run = GlueJobBlock(
        job_name="data-processing",
        arguments={"--stockcode":stockcode},
    ).trigger()

    if isinstance(glue_job_run, GlueJobRun): return glue_job_run.fetch_result()
    else: main_logger.error('failed to return gluejobrun object'); return None


@task(name="Run AWS Glue Job")
def process_data(stockcode: str = ''):
    """
    This task simulates the AWS Glue job.
    In a real-world scenario, you would use a Prefect AWS integration
    to trigger and monitor the Glue job, like so:

    `run_glue_job(job_name="my-data-processing-job", ...)`

    For this example, we'll directly call your main script logic.
    """

    if stockcode:
        X_train, X_test, y_train, y_test, preprocessor = dh.main_script_by_stockcode(stockcode)
        return X_train, X_test, y_train, y_test, preprocessor
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = dh.main_script()
        return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


@task(name="Validate Data with Glue Data Quality")
def validate_data_quality(wait_for: list = list()):
    """
    This task represents the data validation step using AWS Glue Data Quality.
    In a real pipeline, it would trigger a Glue Data Quality check and wait for the result.
    For this example, we'll just print a success message.
    """
    print("Running AWS Glue Data Quality checks...")

    if wait_for:
        print()
    # Add your validation logic here, e.g., using a library or API call
    # if not validation_successful:
    #     raise ValueError("Data validation failed! Check Glue Data Quality reports.")
    print("Data validation successful. Data is ready for the next stage.")
    return True

@task(name="Version Data with lakeFS")
def version_data_with_lakefs_task(branch_name: str, commit_message: str):
    """
    This task handles data versioning by interacting with lakeFS.
    It simulates branching and committing the processed data.
    """
    print(f"Creating a new lakeFS branch '{branch_name}' and committing data...")
    # In a real scenario, you'd use a lakeFS SDK or Prefect integration
    # to create a branch, upload files, and commit changes.
    # For instance:
    # lakefs_client.create_branch(repo_id="my-data-lake", name=branch_name)
    # lakefs_client.commit(repo_id="my-data-lake", branch_name=branch_name, message=commit_message)
    print(f"Data committed to lakeFS with message: '{commit_message}'")
    return True

# --- Main Flow Definition ---

@flow(name="Data ETL Pipeline")
def data_etl_pipeline(stockcode: str = '', commit_message: str = "processed data for analysis"):
    """
    The main Prefect flow that orchestrates the entire data pipeline.
    It launches the data processing, runs validation, and versions the data.
    """
    # 1. Data Processing with AWS Glue
    # We use a placeholder task that calls your script.
    # The `wait_for` dependency ensures this task runs first.
    processed_data = process_data(stockcode=stockcode)

    # 2. Data Validation
    # We call the validation task. `wait_for` is optional here as Prefect
    # will automatically handle the dependency, but it's good practice
    # for clarity.
    validation_status = validate_data_quality(wait_for=[processed_data])

    # 3. Data Versioning
    # This task is also dependent on the processing task completing successfully.
    versioning_status = version_data_with_lakefs_task(
        branch_name=f"processed-data-{prefect.context.get_run_id()}",
        commit_message=commit_message,
        wait_for=[processed_data, validation_status]
    )

    print("Pipeline finished successfully!")

# --- Deployment Block ---

if __name__ == "__main__":
    # Create a deployment to make the flow runnable on a schedule or via a trigger.
    # This is what you would register with your Prefect server.
    data_etl_deployment = Deployment.build_from_flow(
        flow=data_etl_pipeline,
        name="data-etl-pipeline-deployment",
        parameters={"stockcode": "21843", "commit_message": "Monthly data refresh"},
        description="Deploys the ETL pipeline to process and version data.",
        tags=["etl", "data-engineering", "aws-glue"]
    )

    # This serves the deployment for local testing.
    # In a production environment, you would use `prefect deploy`
    # to register the deployment with your Prefect server.
    data_etl_deployment.serve(
        # The name of the work pool you want to use
        # This work pool should be configured with an AWS Glue agent or similar
        # to execute your tasks.
        work_pool_name="default-agent-pool"
    )
