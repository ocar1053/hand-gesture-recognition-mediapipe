import json


class RosbridgePublisher:
    def __init__(self,
                 host='192.168.75.29',
                 port=9090,
                 topic='/mediapipe/hands'):
        try:
            import roslibpy  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                'roslibpy is required. Install it with: pip install roslibpy'
            ) from exc

        self._roslibpy = roslibpy
        try:
            self._client = roslibpy.Ros(host=host, port=port)
            self._client.run()
        except Exception as exc:
            raise RuntimeError(
                f'Could not connect to rosbridge at ws://{host}:{port}: {exc}'
            ) from exc

        if not self._client.is_connected:
            raise RuntimeError(
                f'Could not connect to rosbridge at ws://{host}:{port}')

        try:
            self._topic = roslibpy.Topic(self._client, topic, 'std_msgs/String')
            self._topic.advertise()
        except Exception as exc:
            self._client.terminate()
            raise RuntimeError(
                f'Connected to rosbridge but failed to advertise topic {topic}: {exc}'
            ) from exc

    def publish(self, payload):
        self._topic.publish(
            self._roslibpy.Message({'data': json.dumps(payload)}))

    def close(self):
        try:
            self._topic.unadvertise()
        except Exception:
            pass

        try:
            self._client.terminate()
        except Exception:
            pass