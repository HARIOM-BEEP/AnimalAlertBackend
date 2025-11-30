# Configuration file
import os

class Config:
    # Firebase Cloud Messaging (FCM) configuration
    # We'll set this up later for push notifications
    FCM_SERVER_KEY = os.getenv('FCM_SERVER_KEY', 'your_fcm_server_key_here')
    
    # Camera registration
    CAMERA_SECRET = os.getenv('CAMERA_SECRET', 'camera_secret_123')
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.7
    DANGEROUS_ANIMALS = ['lion', 'tiger', 'leopard', 'bear', 'elephant', 'rhino', 'wolf']