# Terraform

Terraform is used to create and manage the AWS infrastructure.
The Terraform configuration files are located in the [terraform folder](../terraform/).

There are several files:
* [main.tf](../terraform/main.tf) - the main configuration file;
* [vars.tf](../terraform/vars.tf) - holds the main variables;
* [.terraform-version](../terraform/.terraform-version) - holds the Terraform version used;
* [modules/s3](../terraform/modules/s3/) - holds information regarding the S3 bucket that holds the MLFlow information.

Tfstate is stored online. For that, you need to create an AWS S3 bucket that can hold it.
This bucket is then defined as the backend "s3" [here](../terraform/main.tf).

How to run:
1. Run `terraform init` command to initialize the configuration;
2. Use `terraform plan` to compare previous local changes with a remote state;
3. Apply the changes to the cloud with `terraform apply`. Confirm the chances to be made with a 'yes'.

For removing the infrastructure from AWS:
* Run `terraform destroy`.