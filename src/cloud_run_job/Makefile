# Makefile for deploying the Metaculus Forecaster Job

-include ../../variables.mk

ROOT_DIR = ../../
UPLOAD_DIR = upload
JOB_NAME = metaculus-forecaster-job

ENV_VARS = PROJECT_ID=$(CLOUD_PROJECT), GCP_BUCKET_NAME=$(GCP_BUCKET_NAME)

ifdef FORECAST_FOLDER
ENV_VARS := $(ENV_VARS),FORECAST_FOLDER=$(FORECAST_FOLDER)
endif
ifdef ASKNEWS_CACHE
ENV_VARS := $(ENV_VARS),ASKNEWS_CACHE=$(ASKNEWS_CACHE)
endif

.PHONY : all clean deploy

all :
	$(MAKE) clean
	$(MAKE) deploy

# just in case
check :
	@if [ -z "$(CLOUD_PROJECT)" ]; then echo "Missing CLOUD_PROJECT in variables.mk"; exit 1; fi
	@if [ -z "$(CLOUD_DEPLOY_REGION)" ]; then echo "Missing CLOUD_DEPLOY_REGION in variables.mk"; exit 1; fi
	@if [ -z "$(WORKFLOW_SERVICE_ACCOUNT)" ]; then echo "Missing WORKFLOW_SERVICE_ACCOUNT in variables.mk"; exit 1; fi
	@if [ -z "$(GCP_BUCKET_NAME)" ]; then echo "Missing GCP_BUCKET_NAME in variables.mk"; exit 1; fi


deploy : check
	mkdir -p $(UPLOAD_DIR)
	cp src/cloud_run_job/main.py $(UPLOAD_DIR)/
	cp -r $(ROOT_DIR)src $(UPLOAD_DIR)/
	cp -r $(ROOT_DIR)Bot $(UPLOAD_DIR)/
	cp $(ROOT_DIR)requirements.txt $(UPLOAD_DIR)/

	gcloud run jobs deploy \
		$(JOB_NAME) \
		--project $(CLOUD_PROJECT) \
		--region $(CLOUD_DEPLOY_REGION) \
		--source $(UPLOAD_DIR) \
		--task-timeout 1h \
		--memory 2Gi \
		--service-account $(WORKFLOW_SERVICE_ACCOUNT) \
		--set-env-vars "$(ENV_VARS)"

clean :
	rm -rf $(UPLOAD_DIR)