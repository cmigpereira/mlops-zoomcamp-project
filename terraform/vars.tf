variable "aws_region" {
  description = "AWS region to create resources"
  default     = "eu-west-2"
}

variable "project_id" {
  description = "project_id"
  default = "mlops-zoomcamp-cpereira"
}

variable "model_bucket" {
  description = "s3_bucket"
  default = "mlflow-bucket-cpereira"
}