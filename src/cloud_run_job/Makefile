# Makefile for deploying the Cloud Run Job

ROOT_DIR = ../..
include $(ROOT_DIR)/variables.mk
UPLOAD_DIR = upload
JOB_NAME = metaculus-forecaster-job

ENV_VARS = PROJECT_ID=$(CLOUD_PROJECT), GCP_BUCKET_NAME=$(GCP_BUCKET_NAME)

ifdef FORECAST_FOLDER
ENV_VARS := $(ENV_VARS),FORECAST_FOLDER=$(FORECAST_FOLDER)
endif

.PHONY : all clean deploy

all :
	$(MAKE) clean
	$(MAKE) check
	$(MAKE) deploy

# just in case
check :
	@if [ -z "$(CLOUD_PROJECT)" ]; then echo "Missing CLOUD_PROJECT in variables.mk"; exit 1; fi
	@if [ -z "$(CLOUD_DEPLOY_REGION)" ]; then echo "Missing CLOUD_DEPLOY_REGION in variables.mk"; exit 1; fi
	@if [ -z "$(WORKFLOW_SERVICE_ACCOUNT)" ]; then echo "Missing WORKFLOW_SERVICE_ACCOUNT in variables.mk"; exit 1; fi
	@if [ -z "$(GCP_BUCKET_NAME)" ]; then echo "Missing GCP_BUCKET_NAME in variables.mk"; exit 1; fi


deploy : main.py Procfile
	$(MAKE) check
	mkdir -p $(UPLOAD_DIR)/src
	cp $^ $(UPLOAD_DIR)/
	cp -r $(ROOT_DIR)/src/helpers $(UPLOAD_DIR)/src
	cp -r $(ROOT_DIR)/Bot $(UPLOAD_DIR)/
	cp $(ROOT_DIR)/requirements.txt $(UPLOAD_DIR)/

	gcloud run jobs deploy $(JOB_NAME) \
		--project $(CLOUD_PROJECT) \
		--region $(CLOUD_DEPLOY_REGION) \
		--source $(UPLOAD_DIR) \
		--tasks 1 \
		--parallelism 1 \
		--task-timeout 24h \
		--memory 2Gi \
		--max-retries 0 \
		--service-account $(WORKFLOW_SERVICE_ACCOUNT) \
		--set-env-vars "$(ENV_VARS)" \

clean :
	rm -rf $(UPLOAD_DIR)
