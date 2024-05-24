# Duplicates Classifier API

## Docker deployment

Run the following bash commands to deploy the ML model in a Docker container.

- Build the image: ```docker buildx build --tag 'duplicates_classifier' .```
- Run the image: ```docker run -it -p 8000:8000 'duplicates_classifier'```

## Kubernetes deployment

Run the following bash commands to deploy the ML model in a Kubernetes cluster.

- Install the kuberay-operator: ```helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0```
- Deploy the ML model: ```kubectl apply -f ray-service.duplicates-classifier.yaml```

### Set up the Kubernetes cluster on GCP

- Create a new project
- Go to *Kubernetes engine*
- Create a cluster:
    - ```Name: lw-duplicates-classifier-cluster```
    - ```Region: europe-southwest1```
- Click on ```lw-duplicates-classifier-cluster```
- Connect (running in a shell): ```gcloud container clusters get-credentials lw-duplicates-classifier-cluster --region europe-southwest1 --project lw-duplicates-classifier```
- Get the yaml file: ```curl -LO https://raw.githubusercontent.com/javiro/kuberay_tests/main/ray-service.duplicates-classifier.yaml```
- Install kuberay-operator: ```helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0```
- Port forward: ```kubectl port-forward svc/rayservice-duplicates-classifier-head-svc 8000:8000```
- Deploy the model: ```kubectl apply -f ray-service.duplicates-classifier.yaml```
- Test it in a new shell via ```ipython```: 

```python
import requests
english_text = "Tomorrow will rain."
response = requests.post("http://localhost:8000", json=english_text)
french_text = response.text
print(french_text)
```
- Clean it:

```sh
kubectl delete -f ray-service.duplicates-classifier.yaml
helm uninstall kuberay-operator
```

- Delete the cluster
