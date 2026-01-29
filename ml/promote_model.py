# promote_model.py
"""
Script to manage model promotion in MLflow Model Registry.
Promotes models from Staging to Production or transitions between stages.
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "diabetes-prediction-model")


def get_client():
    """Get MLflow client with tracking URI configured."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def list_model_versions(model_name: str = MODEL_NAME):
    """List all versions of a registered model."""
    client = get_client()
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            logger.info(f"No versions found for model '{model_name}'")
            return []
        
        logger.info(f"\nModel: {model_name}")
        logger.info("-" * 60)
        
        for v in versions:
            logger.info(f"Version: {v.version} | Stage: {v.current_stage} | Status: {v.status}")
            logger.info(f"  Run ID: {v.run_id}")
            logger.info(f"  Created: {v.creation_timestamp}")
            logger.info("")
        
        return versions
        
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
        return []


def promote_to_staging(model_name: str = MODEL_NAME, version: int = None):
    """
    Promote a model version to Staging.
    If no version specified, promotes the latest version.
    """
    client = get_client()
    
    try:
        if version is None:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.error(f"No versions found for model '{model_name}'")
                return False
            version = max(int(v.version) for v in versions)
        
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Staging",
            archive_existing_versions=False
        )
        
        logger.info(f"✅ Model '{model_name}' version {version} promoted to Staging")
        return True
        
    except Exception as e:
        logger.error(f"Error promoting to Staging: {e}")
        return False


def promote_to_production(model_name: str = MODEL_NAME, version: int = None):
    """
    Promote a model version to Production.
    If no version specified, promotes the latest Staging version.
    Archives existing Production versions.
    """
    client = get_client()
    
    try:
        if version is None:
            versions = client.search_model_versions(f"name='{model_name}'")
            staging_versions = [v for v in versions if v.current_stage == "Staging"]
            
            if not staging_versions:
                logger.error(f"No Staging versions found for model '{model_name}'")
                logger.info("Tip: First promote a version to Staging using --to-staging")
                return False
            
            version = max(int(v.version) for v in staging_versions)
        
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.current_stage == "Production":
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived"
                )
                logger.info(f"Archived previous Production version {v.version}")
        
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Production",
            archive_existing_versions=False
        )
        
        logger.info(f"✅ Model '{model_name}' version {version} promoted to Production")
        return True
        
    except Exception as e:
        logger.error(f"Error promoting to Production: {e}")
        return False


def get_production_model(model_name: str = MODEL_NAME):
    """Get the current Production model version."""
    client = get_client()
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        
        if not prod_versions:
            logger.info(f"No Production version found for model '{model_name}'")
            return None
        
        prod_version = prod_versions[0]
        logger.info(f"Production model: {model_name} v{prod_version.version}")
        logger.info(f"  Run ID: {prod_version.run_id}")
        
        return prod_version
        
    except Exception as e:
        logger.error(f"Error getting Production model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="MLflow Model Registry Management")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Name of the registered model")
    parser.add_argument("--version", type=int, help="Specific version to promote")
    parser.add_argument("--list", action="store_true", help="List all model versions")
    parser.add_argument("--to-staging", action="store_true", help="Promote model to Staging")
    parser.add_argument("--to-production", action="store_true", help="Promote model to Production")
    parser.add_argument("--get-production", action="store_true", help="Get current Production model")
    
    args = parser.parse_args()
    
    if args.list:
        list_model_versions(args.model_name)
    elif args.to_staging:
        promote_to_staging(args.model_name, args.version)
    elif args.to_production:
        promote_to_production(args.model_name, args.version)
    elif args.get_production:
        get_production_model(args.model_name)
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Examples:")
        print("  python promote_model.py --list")
        print("  python promote_model.py --to-staging --version 1")
        print("  python promote_model.py --to-production")
        print("  python promote_model.py --get-production")


if __name__ == "__main__":
    main()
