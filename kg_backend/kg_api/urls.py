from django.urls import path
from .views import api

# The api object is imported from views.py where it was defined as:
# api = NinjaAPI()

urlpatterns = [
    path("api/", api.urls),  # This will mount all API endpoints under /api/
]