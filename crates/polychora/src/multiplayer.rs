use polychora_common::protocol::{ClientMessage, ServerMessage};
use std::io::{self, BufWriter, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;

pub use polychora_common::protocol::{PlayerSnapshot, WorldSnapshotPayload, WorldSummary};

#[derive(Debug)]
pub enum MultiplayerEvent {
    Message(ServerMessage),
    Disconnected(String),
}

pub struct MultiplayerClient {
    server_addr: String,
    outgoing: mpsc::Sender<ClientMessage>,
    incoming: mpsc::Receiver<MultiplayerEvent>,
}

impl MultiplayerClient {
    pub fn connect(server_addr: String, player_name: String) -> io::Result<Self> {
        let stream = TcpStream::connect(&server_addr)?;
        let _ = stream.set_nodelay(true);
        let writer_stream = stream.try_clone()?;

        let (outgoing_tx, outgoing_rx) = mpsc::channel::<ClientMessage>();
        let (incoming_tx, incoming_rx) = mpsc::channel::<MultiplayerEvent>();

        {
            let incoming_tx = incoming_tx.clone();
            thread::spawn(move || {
                let mut writer = BufWriter::new(writer_stream);
                while let Ok(message) = outgoing_rx.recv() {
                    let encoded = match postcard::to_stdvec(&message) {
                        Ok(v) => v,
                        Err(error) => {
                            let _ = incoming_tx.send(MultiplayerEvent::Disconnected(format!(
                                "serialize error: {error}"
                            )));
                            break;
                        }
                    };

                    let len = (encoded.len() as u32).to_le_bytes();
                    if writer.write_all(&len).is_err() {
                        let _ = incoming_tx.send(MultiplayerEvent::Disconnected(
                            "server connection closed while writing".to_string(),
                        ));
                        break;
                    }
                    if writer.write_all(&encoded).is_err() {
                        let _ = incoming_tx.send(MultiplayerEvent::Disconnected(
                            "server connection closed while writing".to_string(),
                        ));
                        break;
                    }
                    if writer.flush().is_err() {
                        let _ = incoming_tx.send(MultiplayerEvent::Disconnected(
                            "server connection closed while flushing".to_string(),
                        ));
                        break;
                    }
                }
            });
        }

        {
            let incoming_tx = incoming_tx.clone();
            thread::spawn(move || {
                let mut reader = stream;
                let mut len_buf = [0u8; 4];
                loop {
                    match reader.read_exact(&mut len_buf) {
                        Ok(()) => {}
                        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                            let _ = incoming_tx.send(MultiplayerEvent::Disconnected(
                                "server connection closed".to_string(),
                            ));
                            break;
                        }
                        Err(error) => {
                            let _ = incoming_tx.send(MultiplayerEvent::Disconnected(format!(
                                "read error: {error}"
                            )));
                            break;
                        }
                    };

                    let len = u32::from_le_bytes(len_buf) as usize;
                    if len > 100_000_000 {
                        let _ = incoming_tx.send(MultiplayerEvent::Disconnected(format!(
                            "oversized message: {} bytes", len
                        )));
                        break;
                    }

                    let mut msg_buf = vec![0u8; len];
                    match reader.read_exact(&mut msg_buf) {
                        Ok(()) => {}
                        Err(error) => {
                            let _ = incoming_tx.send(MultiplayerEvent::Disconnected(format!(
                                "read error: {error}"
                            )));
                            break;
                        }
                    };

                    match postcard::from_bytes::<ServerMessage>(&msg_buf) {
                        Ok(message) => {
                            if incoming_tx
                                .send(MultiplayerEvent::Message(message))
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(error) => {
                            let _ = incoming_tx.send(MultiplayerEvent::Disconnected(format!(
                                "parse error: {error}"
                            )));
                            break;
                        }
                    }
                }
            });
        }

        let client = Self {
            server_addr,
            outgoing: outgoing_tx,
            incoming: incoming_rx,
        };
        client.send(ClientMessage::Hello { name: player_name });
        Ok(client)
    }

    pub fn send(&self, message: ClientMessage) {
        let _ = self.outgoing.send(message);
    }

    pub fn try_recv(&self) -> Option<MultiplayerEvent> {
        self.incoming.try_recv().ok()
    }

    pub fn server_addr(&self) -> &str {
        &self.server_addr
    }
}
