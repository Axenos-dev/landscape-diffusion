from django.urls import path
from . import views

urlpatterns = [
    path("get-images", view=views.GenerateView().generate_images)
]