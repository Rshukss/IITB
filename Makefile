# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DOCKER   ?= docker
MKDIR    ?= mkdir
DIST_DIR ?= $(CURDIR)/dist

include $(CURDIR)/versions.mk

ifeq ($(IMAGE_NAME),)
REGISTRY ?= nvidia
IMAGE_NAME = $(REGISTRY)/k8s-device-plugin
endif

BUILDIMAGE_TAG ?= golang$(GOLANG_VERSION)
BUILDIMAGE ?= $(IMAGE_NAME)-build:$(BUILDIMAGE_TAG)

EXAMPLES := $(patsubst ./examples/%/,%,$(sort $(dir $(wildcard ./examples/*/))))
EXAMPLE_TARGETS := $(patsubst %,example-%, $(EXAMPLES))

CMDS := $(patsubst ./cmd/%/,%,$(sort $(dir $(wildcard ./cmd/*/))))
CMD_TARGETS := $(patsubst %,cmd-%, $(CMDS))

CHECK_TARGETS := assert-fmt vet lint ineffassign misspell
MAKE_TARGETS := binaries build check fmt lint-internal test examples cmds coverage generate $(CHECK_TARGETS)

TARGETS := $(MAKE_TARGETS) $(EXAMPLE_TARGETS) $(CMD_TARGETS)

DOCKER_TARGETS := $(patsubst %,docker-%, $(TARGETS))
.PHONY: $(TARGETS) $(DOCKER_TARGETS)

ifeq ($(VERSION),)
CLI_VERSION = $(LIB_VERSION)$(if $(LIB_TAG),-$(LIB_TAG))
else
CLI_VERSION = $(VERSION)
endif
CLI_VERSION_PACKAGE = github.com/xzaviourr/k8s-device-plugin/internal/info

GOOS ?= linux

binaries: cmds
ifneq ($(PREFIX),)
cmd-%: COMMAND_BUILD_OPTIONS = -o $(PREFIX)/$(*)
endif
cmds: $(CMD_TARGETS)
$(CMD_TARGETS): cmd-%:
	CGO_LDFLAGS_ALLOW='-Wl,--unresolved-symbols=ignore-in-object-files' GOOS=$(GOOS) \
		go build -ldflags "-s -w -X $(CLI_VERSION_PACKAGE).gitCommit=$(GIT_COMMIT) -X $(CLI_VERSION_PACKAGE).version=$(CLI_VERSION)" $(COMMAND_BUILD_OPTIONS) $(MODULE)/cmd/$(*)
build:
	GOOS=$(GOOS) go build ./...

examples: $(EXAMPLE_TARGETS)
$(EXAMPLE_TARGETS): example-%:
	GOOS=$(GOOS) go build ./examples/$(*)

all: check test build binary
check: $(CHECK_TARGETS)

# Apply go fmt to the codebase
fmt:
	go list -f '{{.Dir}}' $(MODULE)/... \
		| xargs gofmt -s -l -w

assert-fmt:
	go list -f '{{.Dir}}' $(MODULE)/... \
		| xargs gofmt -s -l > fmt.out
	@if [ -s fmt.out ]; then \
		echo "\nERROR: The following files are not formatted:\n"; \
		cat fmt.out; \
		rm fmt.out; \
		exit 1; \
	else \
		rm fmt.out; \
	fi

ineffassign:
	ineffassign $(MODULE)/...

lint:
# We use `go list -f '{{.Dir}}' $(MODULE)/...` to skip the `vendor` folder.
	go list -f '{{.Dir}}' $(MODULE)/... | xargs golint -set_exit_status

lint-internal:
# We use `go list -f '{{.Dir}}' $(MODULE)/...` to skip the `vendor` folder.
	go list -f '{{.Dir}}' $(MODULE)/internal/... | xargs golint -set_exit_status

misspell:
	misspell $(MODULE)/...

vet:
	go vet $(MODULE)/...

COVERAGE_FILE := coverage.out
test: build cmds
	go test -v -coverprofile=$(COVERAGE_FILE) $(MODULE)/...

coverage: test
	cat $(COVERAGE_FILE) | grep -v "_mock.go" > $(COVERAGE_FILE).no-mocks
	go tool cover -func=$(COVERAGE_FILE).no-mocks

generate:
	go generate $(MODULE)/...

# Generate an image for containerized builds
# Note: This image is local only
.PHONY: .build-image .pull-build-image .push-build-image
.build-image: docker/Dockerfile.devel
	if [ x"$(SKIP_IMAGE_BUILD)" = x"" ]; then \
		$(DOCKER) build \
			--progress=plain \
			--build-arg GOLANG_VERSION="$(GOLANG_VERSION)" \
			--tag $(BUILDIMAGE) \
			-f $(^) \
			docker; \
	fi

.pull-build-image:
	$(DOCKER) pull $(BUILDIMAGE)

.push-build-image:
	$(DOCKER) push $(BUILDIMAGE)

$(DOCKER_TARGETS): docker-%: .build-image
	@echo "Running 'make $(*)' in docker container $(BUILDIMAGE)"
	$(DOCKER) run \
		--rm \
		-e GOCACHE=/tmp/.cache \
		-v $(PWD):$(PWD) \
		-w $(PWD) \
		--user $$(id -u):$$(id -g) \
		$(BUILDIMAGE) \
			make $(*)

# Start an interactive shell using the development image.
PHONY: .shell
.shell:
	$(DOCKER) run \
		--rm \
		-ti \
		-e GOCACHE=/tmp/.cache \
		-v $(PWD):$(PWD) \
		-w $(PWD) \
		--user $$(id -u):$$(id -g) \
		$(BUILDIMAGE)


### Nebuly targets ###


# Docker image names
PLUGIN_DOCKER_IMAGE_NAME ?= k8s-device-plugin
MPS_SERVER_DOCKER_IMAGE_NAME ?= nvidia-mps-server

# Helm chart URL to push Helm charts
HELM_CHART_REGISTRY ?= oci://ghcr.io/nebuly-ai/helm-charts
# Docker registry
DOCKER_REGISTRY ?= ghcr.io/nebuly-ai


.PHONY: neb-docker-build-plugin
neb-docker-build-plugin:
	sudo docker build -t synergcseiitb/k8s-mps-device-plugin:v1 -f deployments/container/Dockerfile.ubuntu .

.PHONY: neb-docker-build-mps-server
neb-docker-build-mps-server:
	$(DOCKER) build -t ${DOCKER_REGISTRY}/${MPS_SERVER_DOCKER_IMAGE_NAME}:${MPS_SERVER_VERSION} -f deployments/mps-server/Dockerfile deployments/mps-server

.PHONY: neb-docker-push-plugin
neb-docker-push-plugin:
	sudo docker save -o k8s-mps-device-plugin:v1.tar synergcseiitb/k8s-mps-device-plugin:v1
	sudo docker load -i k8s-mps-device-plugin:v1.tar
	sudo docker push synergcseiitb/k8s-mps-device-plugin:v1

.PHONY: neb-docker-push-mps-server
neb-docker-push-mps-server:
	$(DOCKER) push ${DOCKER_REGISTRY}/${MPS_SERVER_DOCKER_IMAGE_NAME}:${MPS_SERVER_VERSION}

.PHONY: neb-docker-build
neb-docker-build: neb-docker-build-plugin neb-docker-build-mps-server

.PHONY: neb-docker-push
neb-docker-push: neb-docker-push-plugin neb-docker-push-mps-server

.PHONY: helm-push
helm-push:
	helm package deployments/helm/nvidia-device-plugin --destination /tmp
	helm push /tmp/nvidia-device-plugin-*.tgz $(HELM_CHART_REGISTRY)
	rm /tmp/nvidia-device-plugin-*.tgz

