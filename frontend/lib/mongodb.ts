import { MongoClient, Db } from 'mongodb'

if (!process.env.DATABASE_URL) {
  throw new Error('Please add your MongoDB URI to .env')
}

const uri = process.env.DATABASE_URL
const options = {}

let client: MongoClient
let clientPromise: Promise<MongoClient>

declare global {
  var _mongoClientPromise: Promise<MongoClient> | undefined
}

if (process.env.NODE_ENV === 'development') {
  if (!global._mongoClientPromise) {
    client = new MongoClient(uri, options)
    global._mongoClientPromise = client.connect()
  }
  clientPromise = global._mongoClientPromise
} else {
  client = new MongoClient(uri, options)
  clientPromise = client.connect()
}

export default clientPromise

export async function getDb(): Promise<Db> {
  const client = await clientPromise
  return client.db('nda-company')
}

