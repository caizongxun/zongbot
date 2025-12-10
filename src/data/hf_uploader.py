"""HuggingFace Hub Integration Module

Handles uploading data and models to HuggingFace Hub.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
from huggingface_hub import HfApi, Repository
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class HFUploader:
    """Upload data and models to HuggingFace Hub."""
    
    def __init__(
        self,
        token: Optional[str] = None,
        data_repo: str = "caizongxun/zongbot-data",
        model_repo: str = "caizongxun/zongbot-models"
    ):
        """Initialize HuggingFace uploader.
        
        Args:
            token: HuggingFace API token. Defaults to HUGGINGFACE_TOKEN env var
            data_repo: Data repository name
            model_repo: Model repository name
        """
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        self.data_repo = data_repo
        self.model_repo = model_repo
        self.api = HfApi(token=self.token)
        
        if not self.token:
            raise ValueError("HuggingFace token not provided. Set HUGGINGFACE_TOKEN env var.")
        
        logger.info(f"HFUploader initialized for repos: {data_repo}, {model_repo}")
    
    def upload_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        folder_path: str = "data",
        commit_message: str = "Update cryptocurrency data"
    ) -> bool:
        """Upload data to HuggingFace.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            folder_path: Folder path in the repo
            commit_message: Git commit message
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Uploading {len(data_dict)} files to {self.data_repo}")
            
            # Upload each dataframe as parquet
            for symbol, df in data_dict.items():
                file_path = f"{folder_path}/{symbol}.parquet"
                
                # Save to temporary location
                temp_path = f"/tmp/{symbol}.parquet"
                df.to_parquet(temp_path, index=False)
                
                # Upload to HF
                self.api.upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=file_path,
                    repo_id=self.data_repo,
                    repo_type="dataset",
                    commit_message=commit_message
                )
                
                os.remove(temp_path)
                logger.info(f"Uploaded {symbol}")
            
            logger.info(f"Successfully uploaded {len(data_dict)} files")
            return True
        
        except Exception as e:
            logger.error(f"Error uploading data: {e}")
            return False
    
    def upload_model(
        self,
        model_path: str,
        commit_message: str = "Update trained model"
    ) -> bool:
        """Upload model to HuggingFace.
        
        Args:
            model_path: Path to model file
            commit_message: Git commit message
        
        Returns:
            True if successful
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            logger.info(f"Uploading model to {self.model_repo}")
            
            file_name = os.path.basename(model_path)
            
            self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=file_name,
                repo_id=self.model_repo,
                repo_type="model",
                commit_message=commit_message
            )
            
            logger.info(f"Successfully uploaded model: {file_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            return False
    
    def download_data(
        self,
        symbol: str,
        local_path: str = "data"
    ) -> Optional[pd.DataFrame]:
        """Download data from HuggingFace.
        
        Args:
            symbol: Cryptocurrency symbol
            local_path: Local path to save data
        
        Returns:
            DataFrame or None if failed
        """
        try:
            file_path = f"{symbol}.parquet"
            
            # Download from HF
            hf_file = self.api.hf_hub_download(
                repo_id=self.data_repo,
                filename=file_path,
                repo_type="dataset"
            )
            
            # Load and save locally
            df = pd.read_parquet(hf_file)
            
            local_file = os.path.join(local_path, file_path)
            os.makedirs(local_path, exist_ok=True)
            df.to_parquet(local_file, index=False)
            
            logger.info(f"Downloaded {symbol} to {local_file}")
            return df
        
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return None
    
    def list_data_files(self) -> list:
        """List all data files in repo.
        
        Returns:
            List of file names
        """
        try:
            repo_info = self.api.repo_info(
                repo_id=self.data_repo,
                repo_type="dataset"
            )
            
            files = [f.filename for f in repo_info.siblings]
            logger.info(f"Found {len(files)} files in data repo")
            return files
        
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
