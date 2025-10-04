import os
import sys
import subprocess
from datetime import timedelta, datetime
from dotenv import load_dotenv

from prefect import flow, task
from prefect.schedules import Schedule
from prefect_aws import AwsCredentials

from src._utils import main_logger


# add project root to the python path - enabling prefect to find the script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@task(retries=3, retry_delay_seconds=30)
def run_dvc_pipeline():
    """prefect task to run dvc pipeline defined in the dvc.yaml file"""

    main_logger.info('... checking dvc status and running pipeline ...')

    # dvc to check all stages on dvc.yaml and run what's needed
    result = subprocess.run(["dvc", "repro"], capture_output=True, text=True, check=True)
    main_logger.info(f'... dvc output:\n{result.stdout}')

    # upload data
    subprocess.run(["dvc", "push"], check=True)



@flow(name="Weekly Data Pipeline")
def weekly_data_flow():
    run_dvc_pipeline()


if __name__ == '__main__':
    load_dotenv(override=True)
    ENV = os.getenv('ENV', 'production')
    DOCKER_HUB_REPO = os.getenv('DOCKER_HUB_REPO')
    ECR_FOR_PREFECT_PATH = os.getenv('ECR_FOR_PREFECT_PATH')
    image_repo = f'{DOCKER_HUB_REPO}:ml-sales-pred-lineage-latest' if ENV == 'local' else f'{ECR_FOR_PREFECT_PATH}:latest'

    weekly_schedule = Schedule(
        interval=timedelta(weeks=1),
        anchor_date=datetime(2025, 9, 29, 9, 0, 0),
        active=True,
    )

    AwsCredentials(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), # type: ignore
        region_name=os.getenv('AWS_REGION_NAME'),
    ).save('aws', overwrite=True)

    weekly_data_flow.deploy(
        name=f'weekly-ml-lineage',
        schedule=weekly_schedule,
        work_pool_name="wp-ml-sales-pred",
        image=image_repo, # create a docker image stored in docker hub (local) or ecr (production)
        concurrency_limit=3,
        push=True
    )
