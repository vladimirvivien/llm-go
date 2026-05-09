.PHONY: build vet test lint fmt all clean forms run demo

GO ?= go
BIN := bin/exception-service
LITERTLM_LIB ?= $(HOME)/include/litertlm/lib
MODEL ?= $(HOME)/models/gemma-4-E2B-it.litertlm

build:
	$(GO) build -o $(BIN) ./cmd/exception-service

vet:
	$(GO) vet ./...

test:
	$(GO) test ./...

lint:
	$(shell $(GO) env GOPATH)/bin/golangci-lint run ./...

fmt:
	gofmt -w .

all: fmt vet build test lint

forms:
	$(GO) run ./tools/synthform -out testdata/forms

clean:
	rm -rf bin data exception.db exception.db-shm exception.db-wal

run: build
	./$(BIN) -lib $(LITERTLM_LIB) -model $(MODEL) -watch watch -data data -db exception.db -addr :8080

demo: build forms
	@echo "Starting service in background, then dropping all 5 forms into watch/."
	@mkdir -p watch
	@./$(BIN) -lib $(LITERTLM_LIB) -model $(MODEL) -watch watch -data data -db exception.db -addr :8080 & echo $$! > .demo.pid
	@sleep 30
	@cp testdata/forms/*.png watch/
	@echo "All 5 forms submitted. Open http://localhost:8080 to watch them flow through."
	@echo "Stop the service with: kill \$$(cat .demo.pid)"
